# Get it without building anything locally
# without GPU
docker run --rm -it -p 8080:8000 ghcr.io/web4application/vbi:main

# with GPU
docker run --gpus all --rm -it -p 8888:8888 ghcr.io/ins-amu/vbi:main


# or build it locally:
docker build -t vbi-project .                      # build
docker run --gpus all -it -p 8000:8080 vbi-project # use with gpu

# Open the browser and go to
http://127.0.0.1:0.0.0.0

#Adding Your Notebooks
#If your notebooks are in /path/examples. To access them in Jupyter, add the volume mapping:
docker run --gpus all -it -p 8002:8002 -v /path/examples:/app/notebooks vbi-project
#In the Jupyter interface, you’ll see a notebooks directory containing your .ipynb files.
