import boto3
import requests
import os
from typing import List, Optional, Tuple, Union
from botocore.exceptions import ClientError
import hashlib
from urllib.parse import urlparse
import shutil
from io import BytesIO
import httpx # Added for async requests
from dotenv import load_dotenv
import logging

# Load environment variables at the top of the file
load_dotenv()

# --- START: Enhanced Logging ---
logger = logging.getLogger(__name__)
# --- END: Enhanced Logging ---

class CloudflareR2Storage:
    def __init__(self):
        self.use_local_fallback = False
        self.r2 = None

        try:
            # Load and validate credentials
            self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
            self.access_key = os.getenv("CLOUDFLARE_ACCESS_KEY_ID")
            self.secret_key = os.getenv("CLOUDFLARE_SECRET_ACCESS_KEY")
            self.bucket_name = os.getenv("CLOUDFLARE_BUCKET_NAME", "ai-agents")

            # Validate credentials
            if not all([self.account_id, self.access_key, self.secret_key]):
                raise ValueError("One or more Cloudflare R2 environment variables are missing.")

            # Initialize R2 client
            endpoint_url = f'https://{self.account_id}.r2.cloudflarestorage.com'
            logger.info(f"Initializing R2 connection to endpoint: {endpoint_url}")
            
            from botocore.config import Config
            session = boto3.session.Session()
            self.r2 = session.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name='auto',
                config=Config(connect_timeout=10, read_timeout=30, retries={'max_attempts': 3})
            )

            logger.info(f"Verifying access to bucket: {self.bucket_name}")
            try:
                self.r2.head_bucket(Bucket=self.bucket_name)
                logger.info("✅ Successfully connected to R2 and verified bucket access")
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code')
                if error_code == '404' or "NotFound" in str(e):
                    logger.warning(f"Bucket '{self.bucket_name}' not found. Attempting to create it.")
                    self.r2.create_bucket(Bucket=self.bucket_name)
                    logger.info(f"✅ Bucket '{self.bucket_name}' created successfully.")
                else:
                    logger.error(f"❌ R2 ClientError during initialization: {e}")
                    self.use_local_fallback = True
        except Exception as e:
            logger.critical(f"❌ Failed to initialize R2: {e}")
            logger.warning("R2 services will not be available. Falling back to local storage for KB documents only.")
            self.use_local_fallback = True
            self.r2 = None

        # Ensure local storage exists for fallback
        if self.use_local_fallback:
            os.makedirs("local_storage/kb", exist_ok=True)

    def test_connection(self):
        """Test the R2 connection and print diagnostic information"""
        if self.use_local_fallback or not self.r2:
            print("R2 is not initialized - using local fallback")
            return False
            
        try:
            print(f"Testing R2 connection...")
            print(f"Account ID: {self.account_id}")
            print(f"Endpoint: https://{self.account_id}.r2.cloudflarestorage.com")
            print(f"Bucket: {self.bucket_name}")
            
            self.r2.head_bucket(Bucket=self.bucket_name)
            print("✅ Connection successful - bucket is accessible")
            return True
        except Exception as e:
            print(f"❌ Connection test failed: {str(e)}")
            return False

    def _ensure_bucket_exists(self) -> None:
        """Ensure the R2 bucket exists. Assumes self.r2 is initialized."""
        if not self.r2:
            raise ConnectionError("R2 client not initialized.")
        try:
            self.r2.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.warning(f"Bucket '{self.bucket_name}' not found. Creating bucket.")
                self.r2.create_bucket(Bucket=self.bucket_name)
                logger.info(f"Bucket '{self.bucket_name}' created successfully.")
            else:
                raise

    def _upload_local_kb(self, file_data: bytes, filename: str) -> Tuple[bool, str]:
        """Upload a knowledge base file (as bytes) to local storage as a fallback."""
        try:
            folder = "kb" 
            local_path = f"local_storage/{folder}/{filename}"
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # file_data is now guaranteed to be bytes from the new workflow
            with open(local_path, 'wb') as f:
                f.write(file_data)
            
            file_url = f"file://{os.path.abspath(local_path)}"
            logger.info(f"✅ Fallback successful: KB file '{filename}' saved locally to '{local_path}'.")
            return True, file_url
        except Exception as e:
            logger.error(f"❌ Error saving KB file '{filename}' locally: {e}")
            return False, str(e)

    def upload_file(self, file_data: bytes, filename: str, is_user_doc: bool = False, 
                    schedule_deletion_hours: int = 72) -> Tuple[bool, str]:
        """
        Uploads file content (as bytes) to R2 with a local fallback for knowledge base files.
        """
        folder = "user_docs" if is_user_doc else "kb"
        key = f"{folder}/{filename}"

        if is_user_doc and (self.use_local_fallback or not self.r2):
            error_msg = f"Cannot upload user document '{filename}'. R2 is not available."
            logger.error(f"CRITICAL: {error_msg}")
            return False, error_msg

        if not self.use_local_fallback and self.r2:
            try:
                logger.info(f"Uploading '{filename}' to R2 key '{key}'...")
                self.r2.upload_fileobj(BytesIO(file_data), self.bucket_name, key)
                file_url = f"https://{self.bucket_name}.{self.account_id}.r2.cloudflarestorage.com/{key}"
                
                # Deletion scheduling logic remains the same
                self.schedule_deletion(key, schedule_deletion_hours)
                
                logger.info(f"✅ File '{filename}' uploaded successfully to R2: {file_url}")
                return True, file_url
            except Exception as e:
                logger.error(f"CRITICAL: R2 upload failed for '{filename}': {e}. Attempting fallback for KB doc.")
                if not is_user_doc:
                    return self._upload_local_kb(file_data, filename)
                return False, f"R2 upload failed for user document: {e}"
        
        # This part is now only for KB docs if R2 failed initially or during the upload attempt
        elif not is_user_doc:
            logger.warning(f"R2 not available, falling back to local storage for KB file '{filename}'.")
            return self._upload_local_kb(file_data, filename)
            
        # Should not be reached for user docs if R2 is down, but as a safeguard:
        return False, "A critical error occurred; could not store the file."

    def get_file_content_bytes(self, key: str) -> Optional[bytes]:
        """Downloads a file's content and returns it as bytes."""
        is_user_doc_key = key.startswith("user_docs/")

        if self.use_local_fallback or not self.r2:
            if is_user_doc_key:
                logger.error(f"R2 is unavailable. Cannot get content for user document '{key}'.")
                return None
            local_source_path = f"local_storage/{key}"
            if os.path.exists(local_source_path):
                try:
                    with open(local_source_path, 'rb') as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"Error reading local fallback file '{key}': {e}")
                    return None
            else:
                logger.error(f"R2 unavailable and KB file '{key}' not found in local fallback.")
                return None
        
        try:
            file_obj = BytesIO()
            self.r2.download_fileobj(self.bucket_name, key, file_obj)
            return file_obj.getvalue()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ['404', 'NoSuchKey']:
                logger.warning(f"File '{key}' not found in R2.")
                if not is_user_doc_key:
                    local_source_path = f"local_storage/{key}"
                    if os.path.exists(local_source_path):
                        logger.info("Found in local fallback, reading content.")
                        with open(local_source_path, 'rb') as f:
                            return f.read()
                return None
            else:
                logger.error(f"R2 ClientError when getting content for '{key}': {e}")
                return None
        except Exception as e:
            logger.error(f"General error getting content for '{key}': {e}")
            return None

    # --- START: REWRITTEN WORKFLOW ---
    def _download_content_from_url(self, url: str) -> Tuple[bool, Union[bytes, str], Optional[str]]:
        """
        Downloads content from a URL.
        Returns: (success, content_bytes_or_error_string, final_filename_or_none)
        """
        try:
            with requests.get(url, timeout=30, stream=True) as response:
                response.raise_for_status()
                
                # Determine filename from URL path
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path)
                if not filename: # If URL is like http://example.com
                    # Try to get extension from content-type
                    content_type = response.headers.get('content-type', '')
                    ext = ".data"
                    if 'pdf' in content_type: ext = '.pdf'
                    elif 'text' in content_type: ext = '.txt'
                    elif 'html' in content_type: ext = '.html'
                    # Use a hash of the URL for a unique name
                    filename = f"{hashlib.md5(url.encode()).hexdigest()}{ext}"

                return True, response.content, filename
        except requests.exceptions.RequestException as e:
            error_message = f"Failed to download content from {url}: {e}"
            logger.error(f"CRITICAL: {error_message}")
            return False, error_message, None

    def download_file_from_url(self, url: str, is_user_doc: bool = False, target_filename: Optional[str] = None) -> Tuple[bool, str]:
        """
        Implements the "Fetch First" workflow.
        Downloads content from a URL and then uploads it to storage.
        Returns (success, stored_file_url_or_error_message).
        """
        # Step 1: Fetch Content First
        logger.info(f"Attempting to download content from URL: {url}")
        success, content_or_error, downloaded_filename = self._download_content_from_url(url)
        
        if not success:
            # The download failed, so we stop immediately.
            return False, str(content_or_error)

        # Use the provided target filename or the one derived from the download
        final_filename = target_filename or downloaded_filename
        if not final_filename: # Safeguard
             final_filename = hashlib.md5(url.encode()).hexdigest() + ".data"
        
        # The content (as bytes) is in content_or_error
        file_content_bytes = content_or_error
        
        # Step 2 & 3: Attempt to Store Content (Upload to R2 with fallback)
        logger.info(f"Content from '{url}' downloaded successfully. Now uploading as '{final_filename}'.")
        return self.upload_file(
            file_data=file_content_bytes, 
            filename=final_filename, 
            is_user_doc=is_user_doc
        )
    # --- END: REWRITTEN WORKFLOW ---
            
    def download_file(self, key: str, local_download_path: str) -> bool:
        """Download a file to a local path, with fallback logic."""
        is_user_doc_key = key.startswith("user_docs/")
        
        # R2 is active, try R2 first
        if not self.use_local_fallback and self.r2:
            try:
                logger.info(f"Attempting to download '{key}' from R2 to '{local_download_path}'...")
                os.makedirs(os.path.dirname(local_download_path), exist_ok=True)
                self.r2.download_file(self.bucket_name, key, local_download_path)
                logger.info(f"File '{key}' downloaded successfully from R2.")
                return True
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") in ['404', 'NoSuchKey']:
                    logger.warning(f"File '{key}' not found in R2.")
                else:
                    logger.error(f"R2 ClientError when downloading '{key}': {e}")
            except Exception as e:
                logger.error(f"General error downloading file '{key}' from R2: {e}")

        # Fallback for KB docs (if R2 failed or was never available)
        if not is_user_doc_key:
            local_source_path = f"local_storage/{key}"
            logger.info(f"Checking local fallback for KB file at '{local_source_path}'...")
            if os.path.exists(local_source_path):
                try:
                    shutil.copy2(local_source_path, local_download_path)
                    logger.info(f"KB file '{key}' downloaded from local fallback to '{local_download_path}'.")
                    return True
                except Exception as e_copy:
                    logger.error(f"Error copying local fallback KB file '{key}': {e_copy}")
                    return False
        
        logger.error(f"Failed to download '{key}' from all available sources.")
        return False
            
    async def download_url_to_temp_file_async(self, url: str, temp_dir: str) -> Tuple[bool, str]:
        """Asynchronously downloads a file from a URL to a temporary local directory."""
        # This method's logic remains sound for its purpose.
        try:
            parsed_url = urlparse(url)
            basename = os.path.basename(parsed_url.path)
            if not basename:
                basename = f"{hashlib.md5(url.encode()).hexdigest()}.data"
            
            local_path = os.path.join(temp_dir, basename)

            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                async with client.stream('GET', url) as response:
                    response.raise_for_status()
                    with open(local_path, 'wb') as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
            
            logger.info(f"Successfully downloaded '{url}' to '{local_path}'")
            return True, os.path.abspath(local_path)
            
        except Exception as e:
            logger.error(f"Error during async download of '{url}': {e}")
            return False, str(e)

    def list_files(self, prefix: str = "") -> List[str]:
        """List files with the given prefix.
        If R2 is active, lists from R2. This won't show KB files that *only* exist locally.
        If R2 initialization failed, lists from local_storage (effectively local_storage/kb/ if prefix allows).
        """
        if self.use_local_fallback or not self.r2:
            print(f"R2 unavailable/uninitialized. Listing files from local_storage with prefix '{prefix}'.")
            try:
                # Normalize prefix for local path construction: remove leading / if any, ensure ends with / if not empty
                local_prefix_dir = prefix
                if local_prefix_dir.startswith('/'): 
                    local_prefix_dir = local_prefix_dir[1:]
                if local_prefix_dir and not local_prefix_dir.endswith('/'):
                    local_prefix_dir += '/'
                
                base_local_dir = "local_storage/"
                # Only allow listing within 'kb/' if prefix specifies it or is empty (implies list all, but we only have kb locally)
                # Or if prefix is 'user_docs/', which should be empty locally.
                
                effective_local_dir = os.path.join(base_local_dir, local_prefix_dir)
                
                # Security/Consistency: If global fallback is on, only list from 'kb' if applicable.
                # User docs should not be listed from local.
                if prefix.startswith("user_docs/"):
                    print("User documents are R2-only; cannot list from local storage.")
                    return []

                # Adjust effective_local_dir if prefix doesn't specify 'kb/' but we are in local fallback for kb
                if not prefix.startswith("kb/") and os.path.exists(os.path.join(base_local_dir, "kb")):
                    # If prefix is generic, list all relevant (i.e. kb) files.
                    # This part of logic might need refinement based on exact desired listing behavior for general prefix.
                    # For now, if prefix is "kb/", it works. If empty, it lists from "local_storage/".
                    pass # current effective_local_dir is okay, or could be forced to "local_storage/kb/"

                listed_files = []
                if os.path.exists(effective_local_dir) and os.path.isdir(effective_local_dir):
                    for f_name in os.listdir(effective_local_dir):
                        if os.path.isfile(os.path.join(effective_local_dir, f_name)):
                            # Return keys relative to bucket (e.g., kb/file.txt)
                            listed_files.append(os.path.join(local_prefix_dir, f_name).replace("\\", "/")) 
                return listed_files
            except Exception as e:
                print(f"Error listing local files with prefix '{prefix}': {e}")
                return []
        
        # R2 is active
        try:
            print(f"Listing files from R2 bucket '{self.bucket_name}' with prefix '{prefix}'.")
            response = self.r2.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except Exception as e:
            print(f"Error listing files from R2 with prefix '{prefix}': {e}")
            return []

    def schedule_deletion(self, key: str, hours: int = 72) -> bool:
        """
        Schedule a file for deletion after specified hours (default: 72 hours)
        This works by setting object metadata with expiration time
        """
        if self.use_local_fallback or not self.r2:
            print(f"R2 unavailable/uninitialized. Cannot schedule deletion for '{key}'.")
            return False
        
        try:
            # First, check if object exists
            try:
                self.r2.head_object(Bucket=self.bucket_name, Key=key)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    print(f"File '{key}' not found in R2, cannot schedule deletion.")
                    return False
                raise
            
            # Set object lifecycle metadata
            import datetime
            
            # Calculate expiration time
            expiration_time = datetime.datetime.now() + datetime.timedelta(hours=hours)
            expiration_timestamp = int(expiration_time.timestamp())
            
            # Copy object to itself with new metadata (can't update metadata directly)
            self.r2.copy_object(
                Bucket=self.bucket_name,
                CopySource={'Bucket': self.bucket_name, 'Key': key},
                Key=key,
                Metadata={
                    'expiration_time': str(expiration_timestamp),
                    'auto_delete': 'true'
                },
                MetadataDirective='REPLACE'
            )
            
            print(f"File '{key}' scheduled for deletion after {hours} hours (at {expiration_time}).")
            return True
        except Exception as e:
            print(f"Error scheduling deletion for '{key}': {e}")
            return False

    def check_and_delete_expired_files(self) -> int:
        """
        Check all files and delete those that have passed their expiration time
        Returns count of deleted files
        """
        if self.use_local_fallback or not self.r2:
            print("R2 unavailable/uninitialized. Cannot check for expired files.")
            return 0
        
        import datetime
        deleted_count = 0
        current_time = datetime.datetime.now().timestamp()
        
        try:
            # List all objects in the bucket
            paginator = self.r2.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    
                    try:
                        # Get object metadata
                        response = self.r2.head_object(
                            Bucket=self.bucket_name,
                            Key=key
                        )
                        
                        metadata = response.get('Metadata', {})
                        if 'expiration_time' in metadata and metadata.get('auto_delete') == 'true':
                            expiration_time = int(metadata['expiration_time'])
                            
                            # Check if file has expired
                            if current_time > expiration_time:
                                # Delete the expired file
                                self.r2.delete_object(
                                    Bucket=self.bucket_name,
                                    Key=key
                                )
                                print(f"Deleted expired file '{key}'")
                                deleted_count += 1
                    except Exception as e_obj:
                        print(f"Error checking metadata for '{key}': {e_obj}")
            
            return deleted_count
        except Exception as e:
            print(f"Error checking for expired files: {e}")
            return 0

    def cleanup_expired_files(self):
        """Run periodic cleanup of expired files"""
        if self.use_local_fallback or not self.r2:
            return
        
        try:
            deleted_count = self.check_and_delete_expired_files()
            if deleted_count > 0:
                print(f"Cleanup completed: deleted {deleted_count} expired files")
        except Exception as e:
            print(f"Error during cleanup of expired files: {e}")