# About the project
This project can be used in various scenarious. For example, rescuers can use it to predict which districts they need to evacuate first of all.
# Prerequisites
For proper functioning you need to preinstall this:
1. Git.
2. Git LFS.
3. Docker.
# Installation and usage:
1. `git clone https://github.com/Supcanc/flood_segmentation`.
2. Make 'test_predictions' folder in this project directory.
3. Add your own images in 'test_images' folder that you want this model to predict.
4. `git lfs install`.
5. `git lfs pull`.
6. `docker build -t flood_segmentation .`.
7. `docker run -p 8000:8000 -v $(pwd)/test_predictions:/flood_segmentation/test_predictions flood_segmentation`.
8. Go to the link `http://0.0.0.0:8000/docs`.
9. Click on 'GET' button.
10. Click on button 'Try it out'.
11. Click button 'Execute' and you'll get your predictions in test_predictions folder.