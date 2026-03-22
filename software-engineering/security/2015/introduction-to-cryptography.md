# Meta Information

- URL: [暗号技術入門 第3版 | SBクリエイティブ](https://www.sbcr.jp/product/4797382228/)
- Support Page: [『暗号技術入門 第3版 秘密の国のアリス』](https://www.hyuki.com/cr/)

---

# 1. Cryptography

## Basic

Cryptography purpose is hiding information from eavesdroppers and then make communication secure between senders and receivers.

The basic idea is to encrypt the message and then send the encrypted message to the receiver. The receiver can decrypt the message and get the original message.

The following provides a confidentiality of the message. However, cryptography can also provide integrity and authentication of the message.

## Basic Algorithm

### Symmetric-key Cryptography

[Symmetric-key cryptography (共通鍵暗号) - MDN Web Docs 用語集 | MDN](https://developer.mozilla.org/ja/docs/Glossary/Symmetric-key_cryptography)

To provide confidentiality.

### Asymmetric-key Cryptography (Public-key Cryptography)

[Public-key cryptography (公開鍵暗号) - MDN Web Docs 用語集 | MDN](https://developer.mozilla.org/ja/docs/Glossary/Public-key_cryptography)

To provide confidentiality, integrity and authentication.

And it is also easy to change the key because the key is public.

### One-way Hash Function

[RFC 3874 - A 224-bit One-way Hash Function: SHA-224 日本語訳](https://tex2e.github.io/rfc-translater/html/rfc3874.html)

To provide integrity.

### Message Authentication Code (MAC)

[Message Authentication Code (MAC)  |  Tink  |  Google for Developers](https://developers.google.com/tink/mac)

[HMAC & Message Authentication Codes (MACs) - Why Hashing alone is NOT ENOUGH for data integrity - YouTube](https://www.youtube.com/watch?v=vdzB5Rraeb4)

To provide authentication and integrity.

### Digital Signature

[Digital Signature  |  Tink  |  Google for Developers](https://developers.google.com/tink/digital-signature)

To avoid spoofing and to provide non-repudiation.

## Others

### JWT (JSON Web Token)

> [!NOTE]
> 本書ではJWTは紹介されていませんが、MACに基づいて作られたトークンであるため、ここで紹介します。

[JSON Web Tokens (JWTs)  |  Tink  |  Google for Developers](https://developers.google.com/tink/jwt)

To provide authentication and integrity.

### OIDC (OpenID Connect)

> [!NOTE]
> 本書ではOIDCは紹介されていませんが、JWTを使用して認証を行うため、ここで紹介します。

[OpenID Connect  |  Sign in with Google  |  Google for Developers](https://developers.google.com/identity/openid-connect/openid-connect)

To provide authentication and authorization.

### PKCE (Proof Key for Code Exchange)

> [!NOTE]
> 本書ではPKCEは紹介されていませんが、OIDCを使用して認証を行う際に使用されるため、ここで紹介します。

[Proof Key for Code Exchange (PKCE)  |  Sign in with Google  |  Google for Developers](https://developers.google.com/identity/protocols/oauth2/native-app#step-1-pkce)

To provide security for native applications when using OIDC.

## Steganography

Steganography is the practice of hiding messages or information within other non-secret text or data.

It is a form of security through obscurity, where the existence of the message is hidden rather than just its content

## History of Cryptography

There are many algorithms in the history of cryptography. Here are some of them:

1. Caeser Cipher: [Caesar cipher - Wikipedia](https://en.wikipedia.org/wiki/Caesar_cipher)
    - It is solved by Brute-force attack because there are only 25 possible keys (shifts).
    - [Brute-force attack - Wikipedia](https://en.wikipedia.org/wiki/Brute-force_attack)
2. Substitution Cipher: [Substitution cipher - Wikipedia](https://en.wikipedia.org/wiki/Substitution_cipher)
    - It is solved by Frequency analysis because each letter is replaced by another letter, and the frequency of letters in the ciphertext can be analyzed to guess the original letters.
    - [Frequency analysis - Wikipedia](https://en.wikipedia.org/wiki/Frequency_analysis)
3. Vigenère Cipher: [Vigenère cipher - Wikipedia](https://en.wikipedia.org/wiki/Vigen%C3%A8re_cipher)
    - This algorithm needs key to encrypt and decrypt the message, and the key is repeated to encrypt the message. Therefore, it is not a simple substitution cipher.
    - It is solved by Kasiski examination because it is a polyalphabetic cipher, and the length of the key can be determined by analyzing the repeated sequences in the ciphertext.
    - [Kasiski examination - Wikipedia](https://en.wikipedia.org/wiki/Kasiski_examination)
4. Enigma Machine: [Enigma machine - Wikipedia](https://en.wikipedia.org/wiki/Enigma_machine)
    - This algorithm needs a machine to encrypt and decrypt the message, and it was used by the Germans during World War II. The machine has a complex wiring system that changes the encryption each time a key is pressed.
    - It is solved by Marian Rejewski because they were able to reconstruct the wiring of the Enigma machine and then use that information to break the code.
    - [Marian Rejewski - Wikipedia](https://en.wikipedia.org/wiki/Marian_Rejewski)
4. Vernam Cipher: [Vernam cipher - Wikipedia](https://en.wikipedia.org/wiki/Vernam_cipher)
    - This algorithm use XOR operation to encrypt and decrypt the message, and it is also known as one-time pad because the key is used only once. It is unbreakable if the key is truly random and kept secret.
5. Symmetric: DES (Data Encryption Standard): [Data Encryption Standard - Wikipedia](https://en.wikipedia.org/wiki/Data_Encryption_Standard)
   - Block cipher that uses a 56-bit key to encrypt 64-bit blocks of data with rounds and different subkeys. It was widely used in the past but is now considered insecure due to its short key length and vulnerability to brute-force attacks.
5. Symmetric: Triple DES (3DES): [Triple DES - Wikipedia](https://en.wikipedia.org/wiki/Triple_DES)
    - An enhancement of DES that applies the DES algorithm three times to each data block, effectively increasing the key length to 168 bits. It was designed to provide better security than DES because the number of blute-force attacks is 2^112, which is much larger than 2^56 for DES. However, it is now considered less secure than AES and is being phased out in favor of AES.
6. Symmetric: AES (Advanced Encryption Standard): [Advanced Encryption Standard - Wikipedia](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard)
    - [FIPS 197, Advanced Encryption Standard (AES)](https://nvlpubs.nist.gov/nistpubs/fips/nist.fips.197.pdf)
    - A block cipher that uses a key size of 128, 192, or 256 bits to encrypt 128-bit blocks of data with rounds and different subkeys. It is currently the most widely used symmetric encryption algorithm and is considered secure for most applications. This is called Rijndael algorithm, and it was selected as the winner of the AES competition held by NIST in 2001.

---

The problem is how to share the key between the sender and receiver in symmetric-key cryptography.

There are four main methods to share the key:
1. Pre-shared key: The sender and receiver agree on a key before communication. This method is not scalable and can be vulnerable to interception during the key exchange process.
2. Key distribution center (KDC): A trusted third party that generates and distributes keys to the sender and receiver. This method can be more secure than pre-shared keys but can still be vulnerable to attacks on the KDC.
3. Diffie-Hellman key exchange: A method that allows the sender and receiver to securely exchange keys over an insecure channel without the need for a trusted third party. It is based on the mathematical problem of discrete logarithms, and it is widely used in many cryptographic protocols, such as SSL/TLS and SSH.
4. Asymmetric-key cryptography: The sender and receiver use a pair of public and private keys to encrypt and decrypt messages. The sender encrypts the message with the receiver's public key, and the receiver decrypts the message with their private key. This method is more secure than the previous methods because the private key is never shared, but it can be slower than symmetric-key cryptography due to the computational complexity of the algorithms used.


---


7. Asymmetric: RSA (Rivest–Shamir–Adleman): [RSA (cryptosystem) - Wikipedia](https://en.wikipedia.org/wiki/RSA_(cryptosystem))
    - It is based on the mathematical problem of factoring large integers, and it is widely used in many cryptographic protocols, such as SSL/TLS and PGP. $\text{Ciphertext} = \text{Plaintext}^e \mod n$ and $\text{Plaintext} = \text{Ciphertext}^d \mod n$, where $n$ is the product of two large prime numbers, $e$ is the public exponent, and $d$ is the private exponent. However, it is vulnerable to certain attacks, such as brute-force attack and man-in-the-middle attack.
    - [man-in-the-middle attack - Wikipedia](https://en.wikipedia.org/wiki/Man-in-the-middle_attack)
    - RSA-OAEP (Optimal Asymmetric Encryption Padding) is a padding scheme that can be used with RSA to provide better security against certain attacks, such as chosen ciphertext attacks. It works by adding random padding to the plaintext before encryption, which makes it more difficult for attackers to guess the original plaintext.
    - [Optimal asymmetric encryption padding - Wikipedia](https://en.wikipedia.org/wiki/Optimal_asymmetric_encryption_padding)
8. Asymmetric: ECC (Elliptic Curve Cryptography): [Elliptic curve cryptography - Wikipedia](https://en.wikipedia.org/wiki/Elliptic_curve_cryptography)
    - It is based on the mathematical problem of elliptic curves, and it is widely used in many cryptographic protocols, such as SSL/TLS and Bitcoin. It provides better security than RSA with smaller key sizes, which makes it more efficient for certain applications, such as mobile devices and IoT devices.

## Block Cipher Modes

1. ECB (Electronic Codebook): [Electronic codebook - Wikipedia](https://en.wikipedia.org/wiki/Electronic_codebook)
    - It is the simplest mode of operation for a block cipher, where each block of plaintext is encrypted independently using the same key. However, it is not recommended for use because it can reveal patterns in the plaintext and is vulnerable to certain attacks.
2. CBC (Cipher Block Chaining): [Cipher block chaining - Wikipedia](https://en.wikipedia.org/wiki/Cipher_block_chaining)
    - It is a mode of operation for a block cipher that uses an initialization vector (IV) to encrypt the first block of plaintext, and then each subsequent block of plaintext is XORed with the previous ciphertext block before being encrypted. This mode provides better security than ECB because it can hide patterns in the plaintext. However, it is still vulnerable to certain attacks, such as padding oracle attacks or bit-flipping attacks for IV.
    - [Padding oracle attack - Wikipedia](https://en.wikipedia.org/wiki/Padding_oracle_attack)
    - [Bit-flipping attack - Wikipedia](https://en.wikipedia.org/wiki/Bit-flipping_attack)
3. CTS (Ciphertext Stealing): [Ciphertext stealing - Wikipedia](https://en.wikipedia.org/wiki/Ciphertext_stealing)
    - It is a mode of operation for a block cipher that allows encryption of data that is not a multiple of the block size without the need for padding. It works by encrypting the last two blocks of plaintext together and then stealing some of the ciphertext to create a final ciphertext that is the same length as the original plaintext. This mode provides better security than ECB and CBC because it can hide patterns in the plaintext and does not require padding, which can be vulnerable to certain attacks.
4. CFB (Cipher Feedback): [Cipher feedback - Wikipedia](https://en.wikipedia.org/wiki/Cipher_feedback)
    - It is a mode of operation for a block cipher that allows encryption of data in units smaller than the block size. It works by encrypting the previous ciphertext block and then XORing it with the current plaintext block to produce the current ciphertext block. This mode provides better security than ECB because it can hide patterns in the plaintext and allows for encryption of data in smaller units. However, it is still vulnerable to certain attacks, such as replay attacks or bit-flipping attacks for IV.
    - [Replay attack - Wikipedia](https://en.wikipedia.org/wiki/Replay_attack)
5. OFB (Output Feedback): [Output feedback - Wikipedia](https://en.wikipedia.org/wiki/Output_feedback)
    - It is a mode of operation for a block cipher that allows encryption of data in units smaller than the block size. It works by encrypting the previous ciphertext block and then XORing it with the current plaintext block to produce the current ciphertext block. This mode provides better security than ECB because it can hide patterns in the plaintext and allows for encryption of data in smaller units. However, it is still vulnerable to certain attacks, such as replay attacks or bit-flipping attacks for IV.
6. CTR (Counter): [Counter mode - Wikipedia](https://en.wikipedia.org/wiki/Counter_mode)
    - It is a mode of operation for a block cipher that allows encryption of data in units smaller than the block size. It works by encrypting a counter value, called a nonce, and then XORing it with the current plaintext block to produce the current ciphertext block. This mode provides better security than ECB because it can hide patterns in the plaintext and allows for encryption of data in smaller units. However, it is still vulnerable to certain attacks, such as replay attacks or bit-flipping attacks for IV.

# 2. Authentication

1. Hash: One-way hash function: [One-way function - Wikipedia](https://en.wikipedia.org/wiki/One-way_function)
    - A one-way hash function is a mathematical function that takes an input and produces a fixed-size output, called a hash value or digest. It is designed to be easy to compute the hash value from the input, but it is computationally infeasible to reverse the process and obtain the original input from the hash value. This property makes it useful for authentication and integrity verification, as it allows us to verify that a message has not been tampered with without revealing the original message.
2. Hash: SHA (Secure Hash Algorithm): [Secure Hash Algorithm - Wikipedia](https://en.wikipedia.org/wiki/Secure_Hash_Algorithm)
    - To guarantee the integrity of the message, we can use a hash function to generate a hash value of the message, and then send the hash value along with the message. The receiver can then generate a hash value of the received message and compare it with the received hash value to verify the integrity of the message. There are different versions of SHA, such as SHA-1, SHA-256, and SHA-3, each with different hash lengths and security properties. However, it is important to note that some versions of SHA, such as SHA-1, are no longer considered secure due to vulnerabilities that have been discovered.
    - [Collision attack - Wikipedia](https://en.wikipedia.org/wiki/Collision_attack)
3. Message Authentication Code (MAC): [Message Authentication Code - Wikipedia](https://en.wikipedia.org/wiki/Message_authentication_code)
    - To provide authentication and integrity, we can use a message authentication code (MAC) to generate a hash value of the message along with a secret key. The sender generates the MAC using the secret key and sends it along with the message. The receiver can then generate the MAC using the same secret key and compare it with the received MAC to verify the authenticity and integrity of the message. There are different types of MACs, such as HMAC (Hash-based Message Authentication Code) and CMAC (Cipher-based Message Authentication Code), each with different security properties.
    - [HMAC - Wikipedia](https://en.wikipedia.org/wiki/HMAC)
    - [CMAC - Wikipedia](https://en.wikipedia.org/wiki/CMAC)
4. Encrypt then MAC: [Encrypt-then-MAC - Wikipedia](https://en.wikipedia.org/wiki/Encrypt-then-MAC)
    - To provide confidentiality, authentication, and integrity, we can use the encrypt-then-MAC approach. In this approach, the sender first encrypts the message using a symmetric encryption algorithm, and then generates a MAC of the ciphertext using a secret key. The sender sends both the ciphertext and the MAC to the receiver. The receiver can then verify the authenticity and integrity of the message by generating a MAC of the received ciphertext using the same secret key and comparing it with the received MAC. If the MACs match, the receiver can then decrypt the ciphertext to obtain the original message. This is more secure than the MAC-then-encrypt approach because it ensures that the integrity of the message is verified before decryption, which can help prevent certain types of attacks, such as padding oracle attacks or bit-flipping attacks for IV.
    - [Padding oracle attack - Wikipedia](https://en.wikipedia.org/wiki/Padding_oracle_attack)
5. AEAD (Authenticated Encryption with Associated Data): [Authenticated encryption - Wikipedia](https://en.wikipedia.org/wiki/Authenticated_encryption)
    - To provide confidentiality, authentication, and integrity, we can use authenticated encryption with associated data (AEAD). In this approach, the sender encrypts the message using a symmetric encryption algorithm and generates a MAC of the ciphertext and any associated data using a secret key. The sender sends both the ciphertext and the MAC to the receiver. The receiver can then verify the authenticity and integrity of the message by generating a MAC of the received ciphertext and associated data using the same secret key and comparing it with the received MAC. If the MACs match, the receiver can then decrypt the ciphertext to obtain the original message. This approach is more secure than encrypt-then-MAC because it allows for additional data to be authenticated along with the message, which can help prevent certain types of attacks, such as replay attacks or bit-flipping attacks for IV.
    - [Replay attack - Wikipedia](https://en.wikipedia.org/wiki/Replay_attack)
    - [Bit-flipping attack - Wikipedia](https://en.wikipedia.org/wiki/Bit-flipping_attack)
6. Digital Signature: [Digital signature - Wikipedia](https://en.wikipedia.org/wiki/Digital_signature)
    - To provide non-repudiation, we can use a digital signature to sign the message using a private key. The sender generates a digital signature of the message using their private key and sends it along with the message. The receiver can then verify the authenticity of the message by generating a digital signature of the received message using the sender's public key and comparing it with the received digital signature. If the signatures match, the receiver can be confident that the message was indeed sent by the sender and has not been tampered with. This provides non-repudiation because the sender cannot deny sending the message since only they have access to their private key. However, it is vulnerable to certain attacks, such as man-in-the-middle attack or key compromise, which can allow an attacker to forge a digital signature or impersonate the sender
    - [Non-repudiation - Wikipedia](https://en.wikipedia.org/wiki/Non-repudiation)

# 3. Key, Random Number and Advanced Technology


# Other Topics

- [CRYPTREC | トップページ](https://www.cryptrec.go.jp/index.html)
    - CRYPTREC (Cryptography Research and Evaluation Committees) is a project that evaluates and recommends cryptographic algorithms for use in Japan. It provides a list of recommended algorithms for different purposes, such as encryption, hashing, and digital signatures.
- [Swift](https://www.swift.com/)
    - Swift is a global provider of secure financial messaging services. It provides a platform for financial institutions to exchange information securely and efficiently. Swift also provides a range of security services, such as encryption and authentication, to protect the confidentiality and integrity of financial messages.
- [IPsec - Wikipedia](https://ja.wikipedia.org/wiki/IPsec)
    - IPsec (Internet Protocol Security) is a suite of protocols that provides secure communication over IP networks. It uses encryption and authentication to protect the confidentiality and integrity of data transmitted over the network. IPsec can be used to create virtual private networks (VPNs) and to secure communication between different networks.
