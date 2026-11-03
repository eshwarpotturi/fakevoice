"""List the files inside the remote archives and fetch the ASVspoof 2021 DF labels, without downloading the audio.
Writes data/wavefake_names.json, data/asv_names.json and data/keys/DF/CM/trial_metadata.txt. Run from the repository root."""
import json, os, tarfile, urllib.request
from remotezip import RemoteZip
os.makedirs("data", exist_ok=True)
json.dump(RemoteZip("https://zenodo.org/api/records/5642694/files/generated_audio.zip/content").namelist(), open("data/wavefake_names.json", "w"))
json.dump(RemoteZip("https://huggingface.co/datasets/Bisher/ASVspoof_2021_DF/resolve/main/ASVspoof_DF_2021.zip").namelist(), open("data/asv_names.json", "w"))
urllib.request.urlretrieve("https://www.asvspoof.org/resources/DF-keys-full.tar.gz", "data/dfkeys.tgz")
with tarfile.open("data/dfkeys.tgz") as t: t.extract("keys/DF/CM/trial_metadata.txt", "data")
print("done")
