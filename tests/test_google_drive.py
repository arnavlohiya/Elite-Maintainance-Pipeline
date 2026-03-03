import unittest
from unittest.mock import MagicMock, patch, mock_open
import sys
import os
from pathlib import Path

# Add the project root to sys.path to allow for absolute imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to be tested
from src.web.backend import backend_demo as backendDemo

class TestGoogleDrive(unittest.TestCase):
    def setUp(self):
        # Create dummy paths and IDs
        self.client_secret = Path("dummy_client_secret.json")
        self.root_folder_id = "dummy_root_id"
        
        # Create a mock service object to be used in tests
        self.mock_service = MagicMock()

    @patch("src.web.backend.backend_demo.build")
    @patch("src.web.backend.backend_demo.Credentials")
    @patch("src.web.backend.backend_demo.InstalledAppFlow")
    @patch("src.web.backend.backend_demo.Request")
    @patch("pathlib.Path.exists")
    def test_init_with_valid_token(self, mock_exists, mock_request, mock_flow, mock_creds, mock_build):
        """Test initialization when a valid token.json exists."""
        # Arrange
        # Simulate token.json exists
        mock_exists.return_value = True
        
        # Mock credentials object
        mock_creds_instance = MagicMock()
        mock_creds_instance.valid = True
        mock_creds.from_authorized_user_file.return_value = mock_creds_instance
        
        # Mock build to return our service mock
        mock_build.return_value = self.mock_service

        # Act
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)

        # Assert
        # Should load from file
        mock_creds.from_authorized_user_file.assert_called()
        # Should not run OAuth flow
        mock_flow.from_client_secrets_file.assert_not_called()
        # Should build service
        mock_build.assert_called_with('drive', 'v3', credentials=mock_creds_instance)
        self.assertEqual(drive.service, self.mock_service)

    @patch("src.web.backend.backend_demo.build")
    @patch("src.web.backend.backend_demo.Credentials")
    @patch("src.web.backend.backend_demo.InstalledAppFlow")
    @patch("src.web.backend.backend_demo.Request")
    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_init_needs_auth_flow(self, mock_file, mock_exists, mock_request, mock_flow, mock_creds, mock_build):
        """Test initialization when token.json is missing (triggers OAuth flow)."""
        # Arrange
        # Simulate token.json does NOT exist, but client_secret DOES exist
        # We use side_effect list: 1st call (token) -> False, 2nd call (secret) -> True
        mock_exists.side_effect = [False, True]

        # Mock Flow
        mock_flow_instance = MagicMock()
        mock_flow.from_client_secrets_file.return_value = mock_flow_instance
        
        mock_creds_instance = MagicMock()
        mock_flow_instance.run_local_server.return_value = mock_creds_instance
        
        mock_build.return_value = self.mock_service

        # Act
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)

        # Assert
        mock_flow.from_client_secrets_file.assert_called_with(str(self.client_secret), backendDemo.SCOPES)
        mock_flow_instance.run_local_server.assert_called()
        mock_build.assert_called_with('drive', 'v3', credentials=mock_creds_instance)
        
        # Should save token (open called for write)
        mock_file.assert_called()
        handle = mock_file()
        handle.write.assert_called()

    def test_get_or_create_folder_existing(self):
        """Test retrieving an existing folder ID."""
        # Arrange
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        drive.service = self.mock_service
        
        # Mock API response for list: returns one folder
        self.mock_service.files().list().execute.return_value = {
            'files': [{'id': 'existing_folder_id', 'name': 'TargetFolder'}]
        }

        # Act
        folder_id = drive._get_or_create_folder("TargetFolder", "parent_id")

        # Assert
        self.assertEqual(folder_id, 'existing_folder_id')
        # Verify cache update
        self.assertEqual(drive.folder_ids_cache[("TargetFolder", "parent_id")], 'existing_folder_id')
        # Verify create was NOT called
        self.mock_service.files().create.assert_not_called()

    def test_get_or_create_folder_create_new(self):
        """Test creating a folder if it doesn't exist."""
        # Arrange
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        drive.service = self.mock_service
        
        # Mock list response (empty)
        self.mock_service.files().list().execute.return_value = {'files': []}
        # Mock create response
        self.mock_service.files().create().execute.return_value = {'id': 'new_folder_id'}

        # Act
        folder_id = drive._get_or_create_folder("NewFolder", "parent_id")

        # Assert
        self.assertEqual(folder_id, 'new_folder_id')
        self.mock_service.files().create.assert_called()
        
        # Verify arguments
        call_args = self.mock_service.files().create.call_args
        self.assertEqual(call_args[1]['body']['name'], 'NewFolder')
        self.assertEqual(call_args[1]['body']['mimeType'], 'application/vnd.google-apps.folder')

    def test_get_unique_name_conflict(self):
        """Test generating a unique filename when conflicts exist."""
        # Arrange
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        drive.service = self.mock_service

        # Mock existing filenames
        self.mock_service.files().list().execute.return_value = {
            'files': [{'name': 'test_file.txt'}, {'name': 'test_file (1).txt'}]
        }
        
        # Reset mock to clear the call made during setup
        self.mock_service.reset_mock()

        # Act
        unique_name = drive._get_unique_name("parent_id", "test_file.txt")

        # Assert
        self.assertEqual(unique_name, "test_file (2).txt")

        # Verify list was called with the correct arguments
        self.mock_service.files().list.assert_called_once_with(
            q="'parent_id' in parents and trashed=false",
            fields="files(name)"
        )

    def test_get_unique_name_no_conflict(self):
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        drive.service = self.mock_service
        
        self.mock_service.files().list().execute.return_value = {'files': []}
        unique_name = drive._get_unique_name("parent_id", "test_file.txt")
        self.assertEqual(unique_name, "test_file.txt")

    @patch("src.web.backend.backend_demo.MediaIoBaseUpload")
    def test_write_manifest(self, mock_media_upload):
        """Test uploading a manifest JSON file."""
        # Arrange
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        drive.service = self.mock_service
        
        # Mock _get_unique_name to avoid complex logic here
        drive._get_unique_name = MagicMock(return_value="manifest.json")
        
        manifest_data = {"job_id": "123", "status": "ok"}

        # Act
        drive.write_manifest(manifest_data, "parent_id")

        # Assert
        self.mock_service.files().create.assert_called()
        call_args = self.mock_service.files().create.call_args
        self.assertEqual(call_args[1]['body']['name'], 'manifest.json')
        self.assertEqual(call_args[1]['body']['parents'], ['parent_id'])
        mock_media_upload.assert_called()

    @patch("src.web.backend.backend_demo.MediaFileUpload")
    def test_push_video(self, mock_media_upload):
        """Test uploading a video file."""
        # Arrange
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        drive.service = self.mock_service
        drive._get_unique_name = MagicMock(return_value="video.360")
        
        self.mock_service.files().create().execute.return_value = {'id': 'video_file_id'}
        video_path = Path("test_video.360")
        canonical_name = "test_video.360"
        
        # Act
        file_id = drive.push_video(video_path, "parent_id", canonical_name)

        # Assert
        self.assertEqual(file_id, 'video_file_id')
        mock_media_upload.assert_called_with(str(video_path), mimetype='application/octet-stream', resumable=True)

    # --- Tests for missing methods ---

    def test_ensure_to_process(self):
        """Test ensure_to_process creates ToProcess/{wbid} folder chain."""
        # Arrange
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        drive.service = self.mock_service
        drive._get_or_create_folder = MagicMock(side_effect=["to_process_root_id", "wbid_folder_id"])
        
        # Act
        result = drive.ensure_to_process("GS012345")
        
        # Assert
        self.assertEqual(result, "wbid_folder_id")
        # Verify folders created in order
        calls = drive._get_or_create_folder.call_args_list
        self.assertEqual(calls[0][0], ("ToProcess", self.root_folder_id))
        self.assertEqual(calls[1][0], ("GS012345", "to_process_root_id"))

    def test_ensure_to_process_service_unavailable(self):
        """Test ensure_to_process returns None when service unavailable."""
        # Arrange
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        drive.service = None
        
        # Act
        result = drive.ensure_to_process("GS012345")
        
        # Assert
        self.assertIsNone(result)

    def test_ensure_processed(self):
        """Test ensure_processed creates Processed/{wbid} folder chain."""
        # Arrange
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        drive.service = self.mock_service
        drive._get_or_create_folder = MagicMock(side_effect=["processed_root_id", "wbid_folder_id"])
        
        # Act
        result = drive.ensure_processed("GS012345")
        
        # Assert
        self.assertEqual(result, "wbid_folder_id")
        # Verify folders created in order
        calls = drive._get_or_create_folder.call_args_list
        self.assertEqual(calls[0][0], ("Processed", self.root_folder_id))
        self.assertEqual(calls[1][0], ("GS012345", "processed_root_id"))

    def test_ensure_processed_service_unavailable(self):
        """Test ensure_processed returns None when service unavailable."""
        # Arrange
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        drive.service = None
        
        # Act
        result = drive.ensure_processed("GS012345")
        
        # Assert
        self.assertIsNone(result)

    @patch("src.web.backend.backend_demo.MediaIoBaseUpload")
    def test_upload_content(self, mock_media_upload):
        """Test uploading bytes content to Google Drive."""
        # Arrange
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        drive.service = self.mock_service
        self.mock_service.files().create().execute.return_value = {'id': 'content_file_id'}
        
        content = b"test content here"
        
        # Act
        file_id = drive.upload_content(content, "test_file.txt", "parent_folder_id", "text/plain")
        
        # Assert
        self.assertEqual(file_id, 'content_file_id')
        mock_media_upload.assert_called_once()
        call_args = self.mock_service.files().create.call_args
        self.assertEqual(call_args[1]['body']['name'], 'test_file.txt')
        self.assertEqual(call_args[1]['body']['parents'], ['parent_folder_id'])

    def test_upload_content_service_unavailable(self):
        """Test upload_content returns None when service unavailable."""
        # Arrange
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        drive.service = None
        
        # Act
        result = drive.upload_content(b"content", "file.txt", "parent_id", "text/plain")
        
        # Assert
        self.assertIsNone(result)

    def test_get_or_create_folder_service_unavailable(self):
        """Test _get_or_create_folder returns None when service unavailable."""
        # Arrange
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        drive.service = None
        
        # Act
        result = drive._get_or_create_folder("FolderName", "parent_id")
        
        # Assert
        self.assertIsNone(result)

    def test_get_or_create_folder_uses_cache(self):
        """Test that _get_or_create_folder returns cached result on second call."""
        # Arrange
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        drive.service = self.mock_service
        
        self.mock_service.files().list().execute.return_value = {
            'files': [{'id': 'folder_id_1'}]
        }
        
        # Act - first call
        result1 = drive._get_or_create_folder("TestFolder", "parent_id")
        
        # Reset mock to verify it's not called again
        self.mock_service.reset_mock()
        
        # Act - second call
        result2 = drive._get_or_create_folder("TestFolder", "parent_id")
        
        # Assert
        self.assertEqual(result1, result2)
        self.assertEqual(result1, 'folder_id_1')
        # Verify list was NOT called on second invocation (cache was used)
        self.mock_service.files().list.assert_not_called()

    def test_push_video_service_unavailable(self):
        """Test push_video returns None when service unavailable."""
        # Arrange
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        drive.service = None
        
        # Act
        result = drive.push_video(Path("video.360"), "parent_id", "video.360")
        
        # Assert
        self.assertIsNone(result)

    def test_write_manifest_service_unavailable(self):
        """Test write_manifest with None parent folder."""
        # Arrange
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        drive.service = self.mock_service
        
        # Act - should not raise, just return silently
        drive.write_manifest({"test": "data"}, None)
        
        # Assert
        self.mock_service.files().create.assert_not_called()

    def test_write_manifest_service_unavailable_no_service(self):
        """Test write_manifest when service is None."""
        # Arrange
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        drive.service = None
        
        # Act - should not raise, just return silently
        drive.write_manifest({"test": "data"}, "parent_id")
        
        # Assert - no exception should be raised
        # This test just ensures graceful handling

    def test_get_unique_name_no_extension(self):
        """Test _get_unique_name with filename lacking extension."""
        # Arrange
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        drive.service = self.mock_service
        
        self.mock_service.files().list().execute.return_value = {
            'files': [{'name': 'filename'}, {'name': 'filename (1)'}]
        }
        
        # Act
        unique_name = drive._get_unique_name("parent_id", "filename")
        
        # Assert
        self.assertEqual(unique_name, "filename (2)")

    def test_get_unique_name_double_extension(self):
        """Test _get_unique_name with double extensions like .tar.gz."""
        # Arrange
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        drive.service = self.mock_service
        
        self.mock_service.files().list().execute.return_value = {
            'files': [{'name': 'archive.tar.gz'}]
        }
        
        # Act
        unique_name = drive._get_unique_name("parent_id", "archive.tar.gz")
        
        # Assert
        # Should only use the last extension (.gz), so result is archive.tar (1).gz
        self.assertEqual(unique_name, "archive.tar (1).gz")

    @patch("src.web.backend.backend_demo.build")
    @patch("src.web.backend.backend_demo.Credentials")
    @patch("src.web.backend.backend_demo.InstalledAppFlow")
    @patch("src.web.backend.backend_demo.Request")
    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_init_credential_refresh(self, mock_file, mock_exists, mock_request, mock_flow, mock_creds, mock_build):
        """Test initialization when credentials exist but are expired with refresh token."""
        # Arrange
        # Simulate token.json exists
        mock_exists.return_value = True
        
        # Mock credentials that are expired but have refresh token
        mock_creds_instance = MagicMock()
        mock_creds_instance.valid = False
        mock_creds_instance.expired = True
        mock_creds_instance.refresh_token = "refresh_token_value"
        mock_creds.from_authorized_user_file.return_value = mock_creds_instance
        
        mock_build.return_value = self.mock_service
        
        # Act
        drive = backendDemo.GoogleDrive(self.client_secret, self.root_folder_id)
        
        # Assert
        # Should have loaded from file
        mock_creds.from_authorized_user_file.assert_called()
        # Should have called refresh
        mock_creds_instance.refresh.assert_called_with(mock_request.return_value)
        # Should not run OAuth flow
        mock_flow.from_client_secrets_file.assert_not_called()
        # Should build service
        mock_build.assert_called_with('drive', 'v3', credentials=mock_creds_instance)


if __name__ == '__main__':
    unittest.main()