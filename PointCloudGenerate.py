import PhotoScan
import os

app = PhotoScan.app

filelist = app.getOpenFileNames("Select Images", "Images(*.jpg)")
chunk = app.document.addChunk()
chunk.addPhotos(filelist)

for camera in chunk.cameras:
    camera.reference.enabled = False

chunk.crs = PhotoScan.CoordinateSystem()

chunk.matchPhotos(accuracy=PhotoScan.HighAccuracy, generic_preselection=True,reference_preselection=False)

chunk.alignCameras()
# PhotoScan.LowQuality
chunk.buildDepthMaps(quality=PhotoScan.HighQuality, filter=PhotoScan.AggressiveFiltering)
chunk.buildDenseCloud()
tempPointCloudPath = os.path.join(os.path.dirname(filelist[0]),"pc.txt")
chunk.exportPoints(tempPointCloudPath, format= PhotoScan.PointsFormat.PointsFormatXYZ, binary=False, precision=6, normals=False, colors=False)

f = open(tempPointCloudPath)
pointcloud = []
for line in f:
    point = list(map(lambda x: float(x), line.split()))
    pointcloud.append(point)
f.close()

imgCameras = chunk.cameras
for i in range(len(imgCameras)):
    camera = imgCameras[i]
    uvPath = os.path.join(os.path.dirname(filelist[0]), camera.label+"_uv.txt")
    f = open(uvPath, 'w')
    for pindex in range(len(pointcloud)):
        point = pointcloud[pindex]
        uv = camera.project(point)
        if uv is not None:
            f.write(str(uv.x)+" "+str(uv.y)+"\n")
        else:
            f.write(str(0) + " " + str(0) + "\n")
    f.close()


