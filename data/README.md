# Dataset directory

Raw data, processed arrays, caches, and participant metadata are intentionally
excluded from Git.

Expected layout:

~~~text
data/
|-- glucose/       # private archive supplied by the corresponding author
|-- phone_camera/  # clone/download of oximetry-phone-cam-data
+-- openox/        # OpenOx repository root or its 1.1.1 directory
~~~

See [../docs/DATASETS.md](../docs/DATASETS.md) for acquisition links and file
expectations.