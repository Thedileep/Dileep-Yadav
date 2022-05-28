from cv2 import line
import pikepdf
from tqdm import tqdm
p= [line.strip() for line in open('wordlist.txt',)]
for p in tqdm(p,'decrypting PDF'):
    try:
        with pikepdf.open('crack.pdf',password=p)as pdf:
            print("\n[+] password found: ",p)
            break
    except pikepdf._qpdf.PasswordError as e:
        continue


