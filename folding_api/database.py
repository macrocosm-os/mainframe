import sqlite3
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import aiosqlite
from loguru import logger


class DatabaseManager:
    def __init__(self, db_path: str = "protein_jobs.db"):
        self.db_path = db_path
    
    async def init_database(self):
        """Initialize the database and create tables if they don't exist"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS protein_jobs (
                    id TEXT NOT NULL PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    pdb_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.commit()
            logger.info("Database initialized successfully")
    
    async def insert_protein_job(
        self, 
        job_id: str, 
        pdb_id: str, 
        user_id: str
    ) -> str:
        """Insert a new protein job record and return the generated ID"""
        record_id = str(uuid.uuid4())
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO protein_jobs (id, job_id, pdb_id, user_id, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (record_id, job_id, pdb_id, user_id, datetime.utcnow().isoformat()))
            await db.commit()
            
        logger.info(f"Inserted protein job: id={record_id}, job_id={job_id}, pdb_id={pdb_id}, user_id={user_id}")
        return record_id
    
    async def get_protein_jobs(
        self, 
        user_id: Optional[str] = None, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get protein jobs, optionally filtered by user_id"""
        async with aiosqlite.connect(self.db_path) as db:
            if user_id:
                cursor = await db.execute("""
                    SELECT id, job_id, pdb_id, user_id, created_at
                    FROM protein_jobs 
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (user_id, limit))
            else:
                cursor = await db.execute("""
                    SELECT id, job_id, pdb_id, user_id, created_at
                    FROM protein_jobs 
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
            
            rows = await cursor.fetchall()
            await cursor.close()
            
            return [
                {
                    "id": row[0],
                    "job_id": row[1], 
                    "pdb_id": row[2],
                    "user_id": row[3],
                    "created_at": row[4]
                }
                for row in rows
            ]
    
    async def get_protein_job_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific protein job by its record ID"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT id, job_id, pdb_id, user_id, created_at
                FROM protein_jobs 
                WHERE id = ?
            """, (record_id,))
            
            row = await cursor.fetchone()
            await cursor.close()
            
            if row:
                return {
                    "id": row[0],
                    "job_id": row[1],
                    "pdb_id": row[2], 
                    "user_id": row[3],
                    "created_at": row[4]
                }
            return None


# Global database manager instance
db_manager = DatabaseManager() 