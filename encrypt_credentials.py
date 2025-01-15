from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os
import getpass

password = getpass.getpass("Enter a password for encryption: ").encode()
salt = os.urandom(16)

kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    length=32,
    salt=salt,
    iterations=100000,
)
key = kdf.derive(password)

def encrypt_data(data):
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    encrypted_data = aesgcm.encrypt(nonce, data.encode(), None)
    return nonce + encrypted_data

openai_api_key = ''
mongo_username = ''
mongo_password = ''

encrypted_openai_api_key = encrypt_data(openai_api_key)
encrypted_mongo_username = encrypt_data(mongo_username)
encrypted_mongo_password = encrypt_data(mongo_password)

with open("encrypted_credentials.txt", "wb") as cred_file:
    cred_file.write(salt + b"\n" + encrypted_openai_api_key + b"\n" + encrypted_mongo_username + b"\n" + encrypted_mongo_password)

print("Credentials encrypted and saved to 'encrypted_credentials.txt'.")
