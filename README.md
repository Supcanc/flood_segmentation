docker build -t flood_segmentation .
docker run -p 8000:8000 -v $(pwd)/test_predictions:/flood_segmentation/test_predictions flood_segmentation