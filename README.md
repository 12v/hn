# HN score prediction

## Setup

1. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

1. Create `database.ini` with the following contents:
    ```ini
    [postgresql]
    dbname=XYZ
    user=XYZ
    password=XYZ
    host=XYZ
    ```
1. Run `python train_embeddings.py`