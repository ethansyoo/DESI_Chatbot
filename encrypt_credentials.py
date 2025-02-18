from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os
import getpass
import hashlib  # ✅ Import for hashing password

# ✅ Ask the user for a password
password = getpass.getpass("Enter a password for encryption: ").encode()
salt = os.urandom(16)  # ✅ Generate a random salt for KDF

# ✅ Derive encryption key from password
kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    length=32,
    salt=salt,
    iterations=100000,
)
key = kdf.derive(password)

# ✅ Store a hash of the password (SHA256, non-reversible)
password_hash = hashlib.sha256(password).hexdigest().encode()

def encrypt_data(data):
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    encrypted_data = aesgcm.encrypt(nonce, data.encode(), None)
    return nonce + encrypted_data

# ✅ Replace credentials with real values
mongo_username = ''
mongo_password = ''
openai_api_key = ''

# ✅ Encrypt credentials
encrypted_openai_api_key = encrypt_data(openai_api_key)
encrypted_mongo_username = encrypt_data(mongo_username)
encrypted_mongo_password = encrypt_data(mongo_password)

# ✅ Store everything in a file
with open("encrypted_credentials.txt", "wb") as cred_file:
    cred_file.write(salt + b"\n" + password_hash + b"\n" + 
                    encrypted_openai_api_key + b"\n" + 
                    encrypted_mongo_username + b"\n" + 
                    encrypted_mongo_password)

print("✅ Credentials encrypted and saved to 'encrypted_credentials.txt'.")
