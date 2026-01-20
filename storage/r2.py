# storage/r2.py
"""
Cloudflare R2 Storage Module for Telegram Publisher

Read-focused module for fetching articles and images from R2 storage,
with archive functionality for moving sent articles.

UNIFIED Folder Structure:
    bucket/
    └── 2026/
        └── January/
            └── Week-4/
                └── 2026-01-20/
                    ├── images/                    # Shared images (all services use this)
                    │   ├── archdaily_001.jpg
                    │   └── dezeen_002.jpg
                    │
                    ├── candidates/                # Pending articles for selection
                    │   ├── manifest.json
                    │   ├── archdaily_001.json
                    │   └── archdaily_002.json
                    │
                    ├── selected/                  # After editorial selection
                    │   └── digest.json
                    │
                    └── archive/                   # Sent articles (JSON only)
                        ├── archdaily_001.json
                        └── dezeen_002.json

Note: Images stay in the shared /images/ folder permanently.
Only JSON files move from candidates/ to archive/ after sending.
"""

import os
import json
from datetime import datetime, date
from typing import Optional, List, Dict
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


class R2Storage:
    """Handles Cloudflare R2 storage operations (read-focused for Telegram)."""

    def __init__(
        self,
        account_id: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        public_url: Optional[str] = None
    ):
        """
        Initialize R2 storage client.

        Args:
            account_id: R2 account ID (or R2_ACCOUNT_ID env var)
            access_key_id: R2 access key (or R2_ACCESS_KEY_ID env var)
            secret_access_key: R2 secret key (or R2_SECRET_ACCESS_KEY env var)
            bucket_name: R2 bucket name (or R2_BUCKET_NAME env var)
            public_url: Public URL for images (or R2_PUBLIC_URL env var)
        """
        self.account_id = account_id or os.getenv("R2_ACCOUNT_ID")
        self.access_key_id = access_key_id or os.getenv("R2_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or os.getenv("R2_SECRET_ACCESS_KEY")
        self.bucket_name = bucket_name or os.getenv("R2_BUCKET_NAME")
        self.public_url = public_url or os.getenv("R2_PUBLIC_URL")

        # Validate required credentials
        missing: List[str] = []
        if not self.account_id:
            missing.append("R2_ACCOUNT_ID")
        if not self.access_key_id:
            missing.append("R2_ACCESS_KEY_ID")
        if not self.secret_access_key:
            missing.append("R2_SECRET_ACCESS_KEY")
        if not self.bucket_name:
            missing.append("R2_BUCKET_NAME")

        if missing:
            raise ValueError(f"Missing R2 credentials: {', '.join(missing)}")

        # Create S3 client configured for R2
        self.client = boto3.client(
            "s3",
            endpoint_url=f"https://{self.account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"}
            )
        )

    # =========================================================================
    # Path Building Utilities
    # =========================================================================

    def _get_week_number(self, dt: date) -> int:
        """Get the week number within the month (1-5)."""
        first_day = dt.replace(day=1)
        day_of_month = dt.day
        first_weekday = first_day.weekday()
        adjusted_day = day_of_month + first_weekday
        week_number = (adjusted_day - 1) // 7 + 1
        return week_number

    def _get_base_path(self, target_date: Optional[date] = None) -> str:
        """
        Get base path for a date.

        Format: YYYY/MonthName/Week-N/YYYY-MM-DD
        """
        if target_date is None:
            target_date = date.today()

        year = target_date.year
        month_name = target_date.strftime("%B")
        week_num = self._get_week_number(target_date)
        date_str = target_date.strftime("%Y-%m-%d")

        return f"{year}/{month_name}/Week-{week_num}/{date_str}"

    def _build_candidate_path(
        self, 
        source_id: str, 
        index: int,
        target_date: Optional[date] = None
    ) -> str:
        """Build path for candidate article JSON."""
        base = self._get_base_path(target_date)
        return f"{base}/candidates/{source_id}_{index:03d}.json"

    def _build_image_path(
        self,
        source_id: str,
        index: int,
        extension: str = "jpg",
        target_date: Optional[date] = None
    ) -> str:
        """
        Build path for article hero image.

        IMPORTANT: Images are in shared /images/ folder at date level.
        Format: YYYY/MonthName/Week-N/YYYY-MM-DD/images/source_NNN.ext
        """
        base = self._get_base_path(target_date)
        return f"{base}/images/{source_id}_{index:03d}.{extension}"

    def _build_manifest_path(self, target_date: Optional[date] = None) -> str:
        """Build path for manifest file."""
        base = self._get_base_path(target_date)
        return f"{base}/candidates/manifest.json"

    def _build_selected_path(self, target_date: Optional[date] = None) -> str:
        """Build path for selected digest."""
        base = self._get_base_path(target_date)
        return f"{base}/selected/digest.json"

    def _build_archive_path(
        self,
        source_id: str,
        index: int,
        target_date: Optional[date] = None
    ) -> str:
        """Build path for archived article JSON."""
        base = self._get_base_path(target_date)
        return f"{base}/archive/{source_id}_{index:03d}.json"

    # =========================================================================
    # Reading Methods
    # =========================================================================

    def get_manifest(self, target_date: Optional[date] = None) -> Optional[dict]:
        """
        Retrieve manifest for a given date.

        Args:
            target_date: Target date (defaults to today)

        Returns:
            Manifest dict or None if not found
        """
        path = self._build_manifest_path(target_date)

        try:
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=path
            )
            content = response["Body"].read().decode("utf-8")
            return json.loads(content)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    def get_candidate(
        self,
        article_id: str,
        target_date: Optional[date] = None
    ) -> Optional[dict]:
        """
        Retrieve a single candidate article.

        Args:
            article_id: Article ID (e.g., "archdaily_001")
            target_date: Target date (defaults to today)

        Returns:
            Candidate dict or None if not found
        """
        # Parse article_id to get source and index
        parts = article_id.rsplit("_", 1)
        if len(parts) != 2:
            return None

        source_id = parts[0]
        try:
            index = int(parts[1])
        except ValueError:
            return None

        path = self._build_candidate_path(source_id, index, target_date)

        try:
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=path
            )
            content = response["Body"].read().decode("utf-8")
            return json.loads(content)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    def get_all_candidates(
        self,
        target_date: Optional[date] = None
    ) -> List[dict]:
        """
        Retrieve all candidate articles for a given date.

        Args:
            target_date: Target date (defaults to today)

        Returns:
            List of candidate dicts
        """
        manifest = self.get_manifest(target_date)
        if not manifest:
            return []

        candidates = []
        for entry in manifest.get("candidates", []):
            article_id = entry.get("id")
            if article_id:
                candidate = self.get_candidate(article_id, target_date)
                if candidate:
                    candidates.append(candidate)

        return candidates

    def get_selected_digest(
        self,
        target_date: Optional[date] = None
    ) -> Optional[dict]:
        """
        Retrieve the selected digest for a given date.

        Args:
            target_date: Target date (defaults to today)

        Returns:
            Digest dict or None if not found
        """
        path = self._build_selected_path(target_date)

        try:
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=path
            )
            content = response["Body"].read().decode("utf-8")
            return json.loads(content)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    def list_dates_with_content(self, year: int, month: int) -> List[date]:
        """
        List all dates that have content for a given month.

        Args:
            year: Year (e.g., 2026)
            month: Month number (1-12)

        Returns:
            List of dates with content
        """
        month_name = date(year, month, 1).strftime("%B")
        prefix = f"{year}/{month_name}/"

        dates_found = set()

        try:
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    # Extract date from path like "2026/January/Week-3/2026-01-20/..."
                    parts = key.split("/")
                    if len(parts) >= 4:
                        date_str = parts[3]  # e.g., "2026-01-20"
                        try:
                            d = date.fromisoformat(date_str)
                            dates_found.add(d)
                        except ValueError:
                            pass
        except ClientError:
            pass

        return sorted(dates_found)

    # =========================================================================
    # Image URL Helper
    # =========================================================================

    def get_image_public_url(self, r2_path: str) -> Optional[str]:
        """
        Get public URL for an image.

        Args:
            r2_path: Path to image in R2

        Returns:
            Public URL or None if no public URL configured
        """
        if not self.public_url or not r2_path:
            return None
        return f"{self.public_url.rstrip('/')}/{r2_path}"

    # =========================================================================
    # Archive Methods (Move sent articles)
    # =========================================================================

    def _copy_object(self, source_key: str, dest_key: str) -> bool:
        """
        Copy an object within the bucket.

        Args:
            source_key: Source object key
            dest_key: Destination object key

        Returns:
            True if successful
        """
        try:
            self.client.copy_object(
                Bucket=self.bucket_name,
                CopySource={"Bucket": self.bucket_name, "Key": source_key},
                Key=dest_key
            )
            return True
        except ClientError as e:
            print(f"   [ERROR] Failed to copy {source_key}: {e}")
            return False

    def _delete_object(self, key: str) -> bool:
        """
        Delete an object from the bucket.

        Args:
            key: Object key to delete

        Returns:
            True if successful
        """
        try:
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=key
            )
            return True
        except ClientError as e:
            print(f"   [ERROR] Failed to delete {key}: {e}")
            return False

    def _move_object(self, source_key: str, dest_key: str) -> bool:
        """
        Move an object (copy then delete).

        Args:
            source_key: Source object key
            dest_key: Destination object key

        Returns:
            True if successful
        """
        if self._copy_object(source_key, dest_key):
            return self._delete_object(source_key)
        return False

    def archive_article(
        self,
        article_id: str,
        target_date: Optional[date] = None
    ) -> bool:
        """
        Move an article JSON from candidates/ to archive/.

        NOTE: Images stay in the shared /images/ folder - they are NOT moved.
        Only the JSON file is archived.

        Args:
            article_id: Article ID (e.g., "archdaily_001")
            target_date: Target date (defaults to today)

        Returns:
            True if article was archived successfully
        """
        if target_date is None:
            target_date = date.today()

        # Parse article_id
        parts = article_id.rsplit("_", 1)
        if len(parts) != 2:
            print(f"   [ERROR] Invalid article_id format: {article_id}")
            return False

        source_id = parts[0]
        try:
            index = int(parts[1])
        except ValueError:
            print(f"   [ERROR] Invalid index in article_id: {article_id}")
            return False

        # Build paths for JSON only (images stay in place)
        candidate_path = self._build_candidate_path(source_id, index, target_date)
        archive_path = self._build_archive_path(source_id, index, target_date)

        # Move the JSON file
        print(f"   [ARCHIVE] Moving {article_id} to archive...")
        return self._move_object(candidate_path, archive_path)

    def archive_articles(
        self,
        article_ids: List[str],
        target_date: Optional[date] = None
    ) -> Dict[str, bool]:
        """
        Archive multiple articles.

        Args:
            article_ids: List of article IDs to archive
            target_date: Target date (defaults to today)

        Returns:
            Dict mapping article_id to success status
        """
        results = {}

        for article_id in article_ids:
            results[article_id] = self.archive_article(article_id, target_date)

        return results

    def update_manifest_after_archive(
        self,
        archived_ids: List[str],
        target_date: Optional[date] = None
    ) -> bool:
        """
        Update the manifest to remove archived articles.

        Args:
            archived_ids: List of article IDs that were archived
            target_date: Target date (defaults to today)

        Returns:
            True if manifest was updated successfully
        """
        manifest = self.get_manifest(target_date)
        if not manifest:
            return False

        # Filter out archived articles
        original_count = len(manifest.get("candidates", []))
        manifest["candidates"] = [
            entry for entry in manifest.get("candidates", [])
            if entry.get("id") not in archived_ids
        ]

        # Update counts
        manifest["total_candidates"] = len(manifest["candidates"])
        manifest["archived_count"] = manifest.get("archived_count", 0) + len(archived_ids)
        manifest["last_archive_time"] = datetime.utcnow().isoformat() + "Z"

        # Save updated manifest
        path = self._build_manifest_path(target_date)
        try:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=path,
                Body=json.dumps(manifest, indent=2),
                ContentType="application/json"
            )
            print(f"   [MANIFEST] Updated: {original_count} -> {manifest['total_candidates']} candidates")
            return True
        except ClientError as e:
            print(f"   [ERROR] Failed to update manifest: {e}")
            return False

    # =========================================================================
    # Connection Testing
    # =========================================================================

    def test_connection(self) -> bool:
        """Test R2 connection and bucket access."""
        try:
            self.client.list_objects_v2(
                Bucket=self.bucket_name,
                MaxKeys=1
            )
            print(f"   [OK] R2 connected: bucket '{self.bucket_name}'")
            if self.public_url:
                print(f"   [OK] Public URL: {self.public_url}")
            return True
        except ClientError as e:
            print(f"   [ERROR] R2 connection failed: {e}")
            return False