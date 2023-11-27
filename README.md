# Visual-Speech-Synthesis-using-ML

install ffmpeg https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/
extract model folder in the app folder
extract data folder in the app folder

create a venv using  `python -m venv venv`


type `venv\scripts\activate` to enter into the venv


run requirements.txt to install all the dependencies


made the code support custom user entered vedios rather then just selecting the dataset ones
still it is using hard coded region of intereset like cropping a specific section of the vedio and expecting the mouth to lie there
doesn't handle the case for when time stemps not equal to 75 made it to handle by dropping if more then 75 and padding with empty frames if less then 75
