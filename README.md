﻿# Image Recognition API using Go and Tensorflow

to run this example run these commands from terminal
```sh
$ git clone https://github.com/aymankh86/image-recognition-api.git
$ cd image-recognition-api
$ docker-compose.exe -f .\docker-compose.yml up --build
```

from another terminal (or request client tool) make a request as following:
```sh
$ curl localhost:8080/recognize -F 'image=@./cat.jpg'
```

response output:
```sh
{
    "filename": "cat.jpg",
    "Labels": [
        {
            "label": "tabby",
            "Probability": 0.30920604
        },
        {
            "label": "tiger cat",
            "Probability": 0.30176574
        },
        {
            "label": "Egyptian cat",
            "Probability": 0.26096028
        },
        {
            "label": "lynx",
            "Probability": 0.11237579
        },
        {
            "label": "Siamese cat",
            "Probability": 0.010782546
        }
    ]
}
```

See [tutorial](https://outcrawl.com/image-recognition-api-go-tensorflow/?utm_source=golangweekly&utm_medium=email)
