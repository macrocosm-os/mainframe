import os
import json
import string
import random
import sqlite3
import requests
from queue import Queue
from typing import List
from dotenv import load_dotenv
from datetime import timezone
from datetime import datetime
import pandas as pd

from atom.epistula.epistula import Epistula
from gjp_models.models import JobBase, SystemConfig, SystemKwargs

load_dotenv()

rqlite_data_dir = os.getenv("RQLITE_DATA_DIR")
db_addr = os.getenv("JOIN_ADDR").split(":")[0] + ":4001"

if rqlite_data_dir is None:
    raise ValueError(
        "RQLITE_DATA_DIR environment variable is not set inside the .env file"
    )
DB_DIR = os.path.abspath(rqlite_data_dir)


class SQLiteJobStore:
    """SQLite-based job store replacing the CSV-based implementation."""

    def __init__(self, db_path=DB_DIR, table_name="jobs"):
        self.db_path = db_path
        self.table_name = table_name
        self.db_file = os.path.join(self.db_path, "db.sqlite")
        self.epistula = Epistula()

    def _row_to_job(self, row) -> "Job":
        """Convert a database row to a Job object."""
        if not row:
            return None

        data = dict(row)
        # Convert stored JSON strings back to Python objects
        data["hotkeys"] = json.loads(data["hotkeys"])
        data["system_config"] = (
            json.loads(data["system_config"]) if data["system_config"] else None
        )
        data["s3_links"] = json.loads(data["s3_links"]) if data["s3_links"] else None
        data["best_cpt_links"] = (
            json.loads(data["best_cpt_links"]) if data["best_cpt_links"] else None
        )
        data["event"] = json.loads(data["event"]) if data["event"] else None
        data["computed_rewards"] = (
            json.loads(data["computed_rewards"]) if data["computed_rewards"] else None
        )

        # Convert timestamps
        for field in ["created_at", "updated_at", "best_loss_at"]:
            if data[field]:
                data[field] = pd.Timestamp(data[field])
            else:
                data[field] = pd.NaT

        # Convert boolean
        data["active"] = bool(data["active"])

        return Job(**data)

    def get_queue(self, validator_hotkey: str, ready=True) -> Queue:
        """
        Get active jobs as a queue that were submitted by the validator_hotkey

        validator_hotkey (str): hotkey of the validator
        ready (bool): pull the data from the pool that are ready for updating based on a datetime filter.
        """

        if ready:
            # Calculate the threshold time for ready jobs
            now = datetime.now(timezone.utc).isoformat()
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE active = 1
                AND datetime(updated_at, '+' || update_interval || ' seconds') <= datetime('{now}')
                AND validator_hotkey = '{validator_hotkey}'
            """
            response = requests.get(
                f"http://{db_addr}/db/query",
                params={"q": query, "level": "strong"},
            )
        else:
            response = requests.get(
                f"http://{db_addr}/db/query",
                params={
                    "q": f"SELECT * FROM {self.table_name} WHERE active = 1 AND validator_hotkey = '{validator_hotkey}'",
                    "level": "strong",
                },
            )

        if response.status_code != 200:
            raise ValueError(f"Failed to get jobs: {response.text}")
        response = response.json()
        if "error" in response.keys():
            raise ValueError(f"Failed to get jobs: {response['error']}")

        response = response["results"][0]

        if "values" not in response.keys():
            return Queue()

        columns = response["columns"]
        values = response["values"]
        rows = [dict(zip(columns, row)) for row in values]

        queue = Queue()
        for row in rows:
            job = self._row_to_job(row)
            queue.put(job)

        return queue

    def get_inactive_queue(self, last_time_checked: str) -> Queue:
        """Get inactive jobs as a queue."""

        # TODO: Implement a way to filter it based on time. We should keep track of the last time
        # we read the db?
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE active = 0
            AND updated_at >= '{last_time_checked}'
            ORDER BY updated_at ASC
        """
        response = requests.get(
            f"http://{db_addr}/db/query",
            params={
                "q": query,
                "consistency": "strong",
            },
        )

        if response.status_code != 200:
            raise ValueError(f"Failed to get jobs: {response.text}")

        response = response.json()["results"][0]

        if "error" in response.keys():
            raise ValueError(f"Failed to get jobs: {response['error']}")
        elif "values" not in response.keys():
            return Queue()

        columns = response["columns"]
        values = response["values"]
        rows = [dict(zip(columns, row)) for row in values]

        queue = Queue()
        for row in rows:
            job = self._row_to_job(row)
            queue.put(job)

        return queue

    def update_gjp_job(self, job: "Job", gjp_address: str, keypair, job_id: str):
        """
        Updates a GJP job with the given parameters.
        Args:
            job (Job): The job object containing job details.
            gjp_address (str): The address of the GJP server.
            keypair (Keypair): The keypair for authentication.
            job_id (str): The ID of the job to be updated.
        Raises:
            ValueError: If the job update fails (response status code is not 200).
        Returns:
            str: The ID of the updated job.
        """

        body = job.model_dump()

        body_bytes = self.epistula.create_message_body(body)
        headers = self.epistula.generate_header(hotkey=keypair, body=body_bytes)

        response = requests.post(
            f"http://{gjp_address}/jobs/update/{job_id}",
            headers=headers,
            data=body_bytes,
        )
        if response.status_code != 200:
            raise ValueError(f"Failed to upload job: {response.text}")
        return response.json()["job_id"]

    # TODO: Find a different way to implement this method
    def get_all_pdbs(self) -> list:
        """
        Retrieve all PDB IDs from the job store.

        Returns:
            list: List of PDB IDs as strings
        """
        response = requests.get(
            f"http://{db_addr}/db/query",
            params={
                "q": f"SELECT pdb_id FROM {self.table_name}",
                "consistency": "strong",
            },
        )

        if response.status_code != 200:
            raise ValueError(f"Failed to get all PDBs: {response.text}")

        response = response.json()["results"][0]

        if "error" in response.keys():
            raise ValueError(f"Failed to get all PDBs: {response['error']}")
        elif "values" not in response.keys():
            return []

        columns = response["columns"]
        values = response["values"]
        rows = [dict(zip(columns, row)) for row in values]

        return [row["pdb_id"] for row in rows]

    def check_for_available_hotkeys(
        self, job: "Job", hotkeys: List[str]
    ) -> (bool, "Job"):
        """Checks the job's hotkeys to only include those that are still valid."""
        job.hotkeys = list(set(job.hotkeys) & set(hotkeys))
        if not job.hotkeys:
            job.active = False
            return False, job
        return True, job

    def __repr__(self):
        """Show current state of the database."""
        with sqlite3.connect(self.db_file) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {self.table_name}", conn)
        return f"{self.__class__.__name__}\n{df.__repr__()}"

    def upload_job(
        self,
        event: dict,
        keypair,
        gjp_address: str,
        **kwargs,
    ):
        """
        Upload a job to the global job pool database.

        Args:
            hotkeys (list): A list of hotkeys.
            keypair (Keypair): The keypair for generating headers.
            gjp_address (str): The address of the api server.
            event (dict): Event data.

        Returns:
            str: The job ID of the uploaded job.

        Raises:
            ValueError: If the job upload fails.
        """
        job = Job(
            pdb_id=event["pdb_id"],
            system_config=SystemConfig(
                ff=event["ff"],
                box=event["box"],
                water=event["water"],
                system_kwargs=SystemKwargs(**event["system_kwargs"]),
            ),
            hotkeys=[""],
            job_type=event["job_type"],
            created_at=pd.Timestamp.now().floor("s"),
            updated_at=pd.Timestamp.now().floor("s"),
            epsilon=event["epsilon"],
            s3_links=event["s3_links"],
            priority=event.get("priority", 1),
            update_interval=event.get(
                "time_to_live", random.randint(7200, 14400)
            ),  # between 2 hours and 4 hours in seconds
            max_time_no_improvement=event.get("max_time_no_improvement", 1),
            is_organic=event.get("is_organic", False),
            job_id=event.get("job_id", None),
            active=event.get("active", True),
            event=event,
            **kwargs,
        )

        body = job.model_dump()

        body_bytes = self.epistula.create_message_body(body)
        headers = self.epistula.generate_header(hotkey=keypair, body=body_bytes)

        response = requests.post(
            f"http://{gjp_address}/jobs", headers=headers, data=body_bytes
        )
        if response.status_code != 200:
            raise ValueError(f"Failed to upload job: {response.text}")
        job.job_id = response.json()["job_id"]
        return job

    async def confirm_upload(self, job_id: str):
        """
        Confirm the upload of a job to the global job pool by trying to read in the uploaded job.

        Args:
            job_id: the job id that you want to confirm is in the pool.

        Returns:
            str: The job ID of the confirmed job.
        """

        response = requests.get(
            f"http://{db_addr}/db/query",
            params={"q": f"SELECT * FROM jobs WHERE job_id = '{job_id}'"},
        )
        if response.status_code != 200:
            raise ValueError(f"Failed to confirm job: {response.text}")

        response = response.json()["results"][0]

        if "error" in response.keys():
            raise ValueError(f"Failed to get all PDBs: {response['error']}")
        elif "values" not in response.keys():
            return None

        columns = response["columns"]
        values = response["values"]
        rows = [dict(zip(columns, row)) for row in values]
        return rows[0]["job_id"]

    async def monitor_db(self):
        """
        Monitor the database for any changes.

        Returns:
            bool: True if the database has changed, False otherwise.
        """
        response = requests.get(f"http://{db_addr}/status?pretty ")
        if response.status_code != 200:
            raise ValueError(f"Failed to monitor db: {response.text}")

        last_log_leader = response.json()["store"]["raft"]["last_log_index"]

        response = requests.get(f"http://{db_addr}/status?pretty ")
        if response.status_code != 200:
            raise ValueError(f"Failed to monitor db: {response.text}")
        last_log_read = response.json()["store"]["raft"]["last_log_index"]

        return (last_log_leader - last_log_read) != 0


class Job(JobBase):
    """Job class for storing job information."""

    async def update(self, loss: float, hotkey: str):
        """Updates the status of a job in the database. If the loss improves, the best loss, hotkey and hashes are updated."""

        self.active = False
        self.best_loss = loss
        self.best_loss_at = pd.Timestamp.now().floor("s")
        self.best_hotkey = hotkey
        self.updated_at = datetime.now(timezone.utc)


class MockJob(Job):
    def __init__(self, n_hotkeys=5, update_seconds=5, stop_after_seconds=3600 * 24):
        self.pdb = self._make_pdb()
        self.ff = "charmm27"
        self.box = "cube"
        self.water = "tip3p"
        self.hotkeys = self._make_hotkeys(n_hotkeys)
        self.created_at = (
            pd.Timestamp.now().floor("s")
            - pd.Timedelta(seconds=random.randint(0, 3600 * 24))
        ).floor("s")
        self.updated_at = self.created_at
        self.best_loss = 0
        self.best_hotkey = random.choice(self.hotkeys)
        self.commit_hash = self._make_commit_hash()
        self.gro_hash = self._make_commit_hash()
        self.update_interval = pd.Timedelta(seconds=update_seconds)
        self.min_updates = 1
        self.max_time_no_improvement = pd.Timedelta(seconds=stop_after_seconds)

    @staticmethod
    def _make_pdb():
        return "".join(random.choices(string.digits + string.ascii_lowercase, k=4))

    @staticmethod
    def _make_hotkeys(n=10, length=8):
        return [MockJob._make_hotkey(length) for _ in range(n)]

    @staticmethod
    def _make_hotkey(length=8):
        return "".join(random.choices(string.digits + string.ascii_letters, k=length))

    @staticmethod
    def _make_commit_hash(k=40):
        return "".join(random.choices(string.digits + string.ascii_letters, k=k))
