import Metashape, os, glob
from PySide2 import QtCore, QtGui, QtWidgets

TYPES = ["TIFF", "TIF", "PNG"]  # Supported input extensions
VIDEO_TYPES = ["AVI", "WMV", "MP4", "MOV", "FLV"]

def get_marker(chunk, label):
    """
    Get marker by name
    """
    for marker in chunk.markers:
        if marker.label.lower() == label.lower():
            return marker
    return None

class AutoprocessDlg(QtWidgets.QDialog):

    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Auto process data")

        self.radioBtn_vid = QtWidgets.QRadioButton("Import video")
        self.radioBtn_img = QtWidgets.QRadioButton("Import photos")
        self.radioBtn_vid.setChecked(False)
        self.radioBtn_img.setChecked(True)

        self.chkMesh = QtWidgets.QCheckBox("Build && Export mesh")
        self.chkTiles = QtWidgets.QCheckBox("Build && Export tiles")

        self.btnQuit = QtWidgets.QPushButton("Close")
        self.btnP1 = QtWidgets.QPushButton("Start")

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.radioBtn_img, 0, 0)
        layout.addWidget(self.radioBtn_vid, 1, 0)
        layout.addWidget(self.chkMesh, 2, 0)
        layout.addWidget(self.chkTiles, 3, 0)
        layout.addWidget(self.btnP1, 4, 0)
        layout.addWidget(self.btnQuit, 4, 1)
        self.setLayout(layout)

        self.btnP1.clicked.connect(self.autoprocess)
        self.btnQuit.clicked.connect(self.reject)

    def autoprocess(self):
        print("Script started...")

        if self.radioBtn_img.isChecked():  # Adding photos
            input_path = Metashape.app.getExistingDirectory("Specify the path to master folder with image folders:")

            if not input_path:
                print("Wrong input path, script aborted.")
                return 0

            export_path = Metashape.app.getExistingDirectory("Specify the path to the export folder:")
            if not export_path:
                print("Wrong export path, script aborted.")
                return 0

            folder_list = [os.path.join(input_path, folder) for folder in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, folder))]

        elif self.radioBtn_vid.isChecked():
            input_path = Metashape.app.getExistingDirectory("Specify the path to master folder with video files:")

            if not input_path:
                print("Wrong input path, script aborted.")
                return 0

            photo_dest_path = Metashape.app.getExistingDirectory("Specify the path to the destination folder for photo files:")

            if not photo_dest_path:
                print("Wrong photo destination path, script aborted.")
                return 0

            export_path = Metashape.app.getExistingDirectory("Specify the path to the export folder:")
            if not export_path:
                print("Wrong export path, script aborted.")
                return 0

            video_list = [file for file in glob.iglob(input_path + "\\*.*", recursive=True) if os.path.isfile(file) and os.path.splitext(file)[1][1:].upper() in VIDEO_TYPES]

            folder_list = []
            for path in video_list:
                frame_folder = os.path.join(photo_dest_path, os.path.basename(path).rsplit(".", 1)[0])
                if not os.path.exists(frame_folder):
                    os.makedirs(frame_folder)
                folder_list.append(frame_folder)
                frame_path = os.path.join(frame_folder, os.path.basename(path).rsplit(".", 1)[0] + "_{filenum}.png")
                doc = Metashape.Document()
                chunk = doc.addChunk()
                chunk.importVideo(path, image_path=frame_path, frame_step=Metashape.FrameStep.MediumFrameStep, custom_frame_step=1, time_start=0, time_end=-1)
                print("Video split into frames: " + path)

        print("Processing started...")
        for path in folder_list:
            photo_list = [os.path.join(path, photo) for photo in os.listdir(path) if photo.upper()[-3:] in TYPES]

            if len(photo_list) < 2:
                print("Not enough images to process in " + path + ", skipping...")
                continue

            label = os.path.basename(path)
            doc = Metashape.Document()
            doc.save(os.path.join(path, label + ".psx"))
            doc = Metashape.app.document
            doc.open(os.path.join(path, label + ".psx"))

            # Creating chunk and loading photos
            chunk = doc.addChunk()
            doc.chunk = chunk
            chunk.label = label
            chunk.addPhotos(photo_list)
            # Setting spherical type for calibration groups
            for sensor in chunk.sensors:
                sensor.type = Metashape.Sensor.Type.Spherical
            doc.save()

            # Align Photos (high accuracy)
            chunk.matchPhotos(downscale=1, generic_preselection=True, reference_preselection=False, filter_stationary_points=True, guided_matching=True, keypoint_limit_per_mpx=1000, tiepoint_limit=4000)
            chunk.alignCameras()
            doc.save()

            # Detect coded targets
            chunk.detectMarkers(target_type=Metashape.CircularTarget12bit, tolerance=50)
            target1 = get_marker(chunk, "target 1")
            target2 = get_marker(chunk, "target 2")
            target3 = get_marker(chunk, "target 3")
            if not target1 or not target2 or not target3:
                print("Not all targets were detected, skipping scaling")
            else:
                scale1 = chunk.addScalebar(target1, target2)
                scale2 = chunk.addScalebar(target2, target3)
                scale1.reference.distance = 0.232
                scale2.reference.distance = 0.173
                scale1.reference.enabled = True
                scale2.reference.enabled = True
                chunk.updateTransform()

            # Resize reconstruction region (bounding box) by 25%
            chunk.resetRegion()
            chunk.region.size = 1.25 * chunk.region.size

            # Build TiledModel from Depth Maps (medium quality)
            chunk.buildDepthMaps(downscale=4, filter_mode=Metashape.MildFiltering)
            doc.save()
            if self.chkTiles.isChecked():
                chunk.buildTiledModel(tile_size=512, source_data=Metashape.DepthMapsData)
                doc.save()

            # Creating export directory by the project name
            project_export_path = os.path.join(export_path, label)
            if not os.path.isdir(project_export_path):
                os.mkdir(project_export_path)

            # Build Model from same depth maps
            if self.chkMesh.isChecked():
                chunk.buildModel(surface_type=Metashape.Arbitrary, source_data=Metashape.DepthMapsData, interpolation=Metashape.EnabledInterpolation, face_count=Metashape.HighFaceCount, vertex_colors=False)
                doc.save()

                # Build texture for mesh model
                chunk.buildUV(mapping_mode=Metashape.GenericMapping, page_count=1, texture_size=8192)
                chunk.buildTexture(blending_mode=Metashape.MosaicBlending, texture_size=8192, fill_holes=True, ghosting_filter=True, transfer_texture=False)
                doc.save()

                # Export mesh model
                chunk.exportModel(os.path.join(project_export_path, label + ".obj"), format=Metashape.ModelFormatOBJ, texture_format=Metashape.ImageFormatJPEG, save_texture=True, save_uv=True, save_normals=False, save_colors=False, save_confidence=False, save_cameras=False, save_markers=False, save_udim=False, save_alpha=False, embed_texture=False, strip_extensions=False)

            # Export tiled model
            if self.chkTiles.isChecked():
                chunk.exportTiledModel(os.path.join(project_export_path, label + ".tls"), format=Metashape.TiledModelFormatTLS)

            print("Completed task for " + path + " directory")

        print("Script finished")
        return 1

def main_batch():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication()
    parent = app.activeWindow()
    dlg = AutoprocessDlg(parent)
    dlg.exec_()

Metashape.app.addMenuItem("Custom Scripts/Batch Autoprocess", main_batch)

print("Script 'Batch Autoprocess' loaded")
