# Medical Data
## Storage
* All data are stored on Google Drive [here](https://drive.google.com/drive/folders/1GmKSN_fxVYphm3LymOJvIsS-XOBw_19T?usp=sharing).

## Data and necessary modifications
* **wiki** - already OK
* **bmc**  - remove dashes
* **rest** - fix dashes

## Run
For medical data preparation, run the `prepare_dataset.py` script in the following way. Parameters are further described directly in the source file or by entering the **-h** or **--help** option.
```bash
python3 prepare_dataset.py [-h] 
        [--action {none,fix,remove}] 
        --data DATA 
        --dest DEST
```