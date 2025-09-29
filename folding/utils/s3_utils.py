import os
import datetime
import boto3
import mimetypes
from typing import Optional, Dict, cast, Any
from abc import ABC, abstractmethod
from botocore.client import Config
from botocore.exceptions import ClientError

from folding.utils.logging import logger

from dotenv import load_dotenv

load_dotenv()


class S3Config:
    """Configuration class for S3 client."""

    def __init__(
        self,
        region_name: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
        miner_bucket_name: str,
    ):
        self.region_name = region_name
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.bucket_name = bucket_name
        self.miner_bucket_name = miner_bucket_name
        self.endpoint_url = f"https://{self.region_name}.digitaloceanspaces.com"

    @classmethod
    def from_env(cls) -> "S3Config":
        """Create S3Config from environment variables."""
        region_name = os.getenv("S3_REGION")
        access_key_id = os.getenv("S3_KEY")
        secret_access_key = os.getenv("S3_SECRET")
        bucket_name = os.getenv("S3_BUCKET")
        miner_bucket_name = os.getenv("S3_MINER_BUCKET")
        if not all(
            [
                region_name,
                access_key_id,
                secret_access_key,
                bucket_name,
                miner_bucket_name,
            ]
        ):
            raise ValueError(
                "Missing required S3 configuration parameters in environment variables"
            )

        return cls(
            region_name=cast(str, region_name),
            access_key_id=cast(str, access_key_id),
            secret_access_key=cast(str, secret_access_key),
            bucket_name=cast(str, bucket_name),
            miner_bucket_name=cast(str, miner_bucket_name),
        )


class BaseHandler(ABC):
    """Abstract base class for handlers that manage content storage operations."""

    @abstractmethod
    def put(self, file_path: str, location: str, **kwargs) -> str:
        """Stores content in a designated storage system."""
        pass

    @abstractmethod
    def get(self, key: str, output_path: str) -> None:
        """Retrieves content from a designated storage system."""
        pass

    @abstractmethod
    def generate_presigned_url(self, key: str, expires_in: int = 7200) -> str:
        """Generates a presigned URL for temporary access to an object."""
        pass


class DigitalOceanS3Handler(BaseHandler):
    """Manages DigitalOcean Spaces S3 operations for file storage."""

    def __init__(
        self,
        config: Optional[S3Config] = None,
        custom_mime_types: Optional[Dict[str, str]] = None,
    ):
        """Initializes a handler for S3 operations with DigitalOcean Spaces.

        Args:
            config (Optional[S3Config]): S3 configuration. If None, will be loaded from env.
            custom_mime_types (Optional[Dict[str, str]]): Custom MIME type mappings.
        """
        self.config = config or S3Config.from_env()
        self.output_url = os.path.join(
            self.config.endpoint_url, self.config.bucket_name
        )

        self.custom_mime_types = custom_mime_types or {
            ".cpt": "application/octet-stream",
            ".pdb": "chemical/x-pdb",
            ".trr": "application/octet-stream",
            ".log": "text/plain",
            ".dcd": "application/octet-stream",
        }

    @property
    def s3_client(self):
        """Creates a configured S3 client on demand."""

        return boto3.client(
            "s3",
            region_name=self.config.region_name,
            endpoint_url=self.config.endpoint_url,
            aws_access_key_id=self.config.access_key_id,
            aws_secret_access_key=self.config.secret_access_key,
            config=Config(signature_version="s3v4"),
        )

    def _get_content_type(
        self, file_path: str, content_type: Optional[str] = None
    ) -> str:
        """Determines the content type for a file."""
        if content_type:
            return content_type

        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()

        return (
            self.custom_mime_types.get(file_ext)
            or mimetypes.guess_type(file_path)[0]
            or "application/octet-stream"
        )

    def put(
        self,
        file_path: str,
        location: str,
        content_type: Optional[str] = None,
        public: bool = False,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Uploads a file to a specified location in the S3 bucket.

        Args:
            file_path (str): Local path to the file to upload.
            location (str): Destination path within the bucket.
            content_type (Optional[str]): MIME type of the file. If None, it's inferred.
            public (bool): Whether to make the file publicly accessible.
            metadata (Optional[Dict[str, str]]): Additional metadata to store with the file.

        Returns:
            str: The S3 key of the uploaded file.

        Raises:
            ClientError: If the upload fails.
        """
        try:
            file_name = os.path.basename(file_path)
            key = os.path.join(location, file_name)

            with open(file_path, "rb") as file:
                data = file.read()

            content_type = self._get_content_type(file_path, content_type)

            extra_args: Dict[str, Any] = {
                "ContentType": content_type,
                "ACL": "public-read" if public else "private",
            }

            if metadata:
                extra_args["Metadata"] = metadata

            self.s3_client.put_object(
                Bucket=self.config.bucket_name, Key=key, Body=data, **extra_args
            )

            return key

        except ClientError as e:
            logger.error(f"S3 upload error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during upload: {e}")
            raise

    def get(self, key: str, output_path: str) -> None:
        """Downloads a file from S3 to the specified output path.

        Args:
            key (str): The S3 key of the file to download.
            output_path (str): Local path where the file should be saved.

        Raises:
            ClientError: If the download fails.
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "wb") as f:
                self.s3_client.download_fileobj(self.config.miner_bucket_name, key, f)

        except ClientError as e:
            logger.error(f"S3 download error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            raise

    def generate_presigned_url(
        self,
        miner_hotkey: str,
        pdb_id: str,
        file_name: str,
        expires_in: int = 7200,
        method: str = "get_object",
    ) -> dict[str, Any]:
        """Generates a presigned URL for temporary access to an object.

        Args:
            key (str): The S3 key of the object.
            expires_in (int): Number of seconds until the URL expires.
            method (str): The S3 operation to allow ('get_object' or 'put_object').

        Returns:
            dict[str, Any]: A presigned URL for accessing the object.

        Raises:
            ClientError: If URL generation fails.
        """
        location = self._get_location(miner_hotkey, pdb_id, file_name)
        content_type = self._get_content_type(file_name)
        try:
            if method == "get_object":
                return self.s3_client.generate_presigned_url(
                    method,
                    Params={"Bucket": self.config.miner_bucket_name, "Key": location},
                )
            elif method == "put_object":
                return self.s3_client.generate_presigned_post(
                    Bucket=self.config.miner_bucket_name,
                    Key=location,
                    Fields={
                        "acl": "private",
                        "Content-Type": content_type,
                    },
                    Conditions=[
                        ["starts-with", "$Content-Type", content_type],
                        {"acl": "private"},
                    ],
                    ExpiresIn=expires_in,
                )
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
            raise

    def _get_location(self, miner_hotkey: str, pdb_id: str, file_name: str) -> str:
        """Get the location of the object in the S3 bucket."""
        return os.path.join(
            "outputs",
            pdb_id,
            miner_hotkey[:8],
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            file_name,
        )
