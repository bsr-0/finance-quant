"""Batch processing for efficient database operations."""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterator, List, Optional, TypeVar

from sqlalchemy import text

from pipeline.db import get_db_manager

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 1000
    flush_interval: int = 10000
    max_retries: int = 3
    retry_delay: float = 1.0


class BatchInserter(Generic[T]):
    """Efficient batch inserter with buffering and retry logic."""
    
    def __init__(
        self,
        table_name: str,
        columns: List[str],
        config: Optional[BatchConfig] = None
    ):
        self.table_name = table_name
        self.columns = columns
        self.config = config or BatchConfig()
        self.buffer: List[dict] = []
        self.total_inserted = 0
        self.db = get_db_manager()
    
    def add(self, record: dict) -> int:
        """Add record to buffer, flush if needed."""
        self.buffer.append(record)
        
        if len(self.buffer) >= self.config.batch_size:
            return self.flush()
        
        return 0
    
    def add_many(self, records: List[dict]) -> int:
        """Add multiple records."""
        inserted = 0
        for record in records:
            inserted += self.add(record)
        return inserted
    
    def flush(self) -> int:
        """Flush buffer to database."""
        if not self.buffer:
            return 0
        
        count = len(self.buffer)
        
        try:
            self._insert_batch(self.buffer)
            self.total_inserted += count
            self.buffer = []
            
            if self.total_inserted % self.config.flush_interval == 0:
                logger.info(f"Inserted {self.total_inserted} rows into {self.table_name}")
            
            return count
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            raise
    
    def _insert_batch(self, records: List[dict]) -> None:
        """Insert batch of records."""
        if not records:
            return
        
        columns_str = ", ".join(self.columns)
        placeholders = ", ".join([f":{col}" for col in self.columns])
        
        sql = text(f"""
            INSERT INTO {self.table_name} ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT DO NOTHING
        """)
        
        with self.db.engine.connect() as conn:
            for record in records:
                # Filter to only include specified columns
                params = {col: record.get(col) for col in self.columns}
                conn.execute(sql, params)
            conn.commit()
    
    def close(self) -> int:
        """Flush remaining records and close."""
        return self.flush()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class StreamingBatchProcessor(Generic[T]):
    """Process large datasets in streaming batches."""
    
    def __init__(
        self,
        processor: Callable[[List[T]], int],
        config: Optional[BatchConfig] = None
    ):
        self.processor = processor
        self.config = config or BatchConfig()
        self.buffer: List[T] = []
        self.total_processed = 0
    
    def process(self, item: T) -> int:
        """Process single item."""
        self.buffer.append(item)
        
        if len(self.buffer) >= self.config.batch_size:
            return self._flush_buffer()
        
        return 0
    
    def _flush_buffer(self) -> int:
        """Flush and process buffer."""
        if not self.buffer:
            return 0
        
        count = self.processor(self.buffer)
        self.total_processed += count
        self.buffer = []
        
        return count
    
    def close(self) -> int:
        """Process remaining items."""
        return self._flush_buffer()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def chunked_iterator(
    items: List[T],
    chunk_size: int
) -> Iterator[List[T]]:
    """Iterate over items in chunks."""
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


class ParallelBatchProcessor:
    """Process batches in parallel using multiple workers."""
    
    def __init__(
        self,
        worker_func: Callable[[List[T]], int],
        num_workers: int = 4,
        batch_size: int = 1000
    ):
        self.worker_func = worker_func
        self.num_workers = num_workers
        self.batch_size = batch_size
    
    def process_all(self, items: List[T]) -> int:
        """Process all items in parallel batches."""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        chunks = list(chunked_iterator(items, self.batch_size))
        total_processed = 0
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self.worker_func, chunk): i
                for i, chunk in enumerate(chunks)
            }
            
            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    count = future.result()
                    total_processed += count
                    logger.info(f"Chunk {chunk_idx}: processed {count} items")
                except Exception as e:
                    logger.error(f"Chunk {chunk_idx} failed: {e}")
        
        return total_processed
