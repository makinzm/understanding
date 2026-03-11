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
5. DES (Data Encryption Standard): [Data Encryption Standard - Wikipedia](https://en.wikipedia.org/wiki/Data_Encryption_Standard)
6. AES (Advanced Encryption Standard): [Advanced Encryption Standard - Wikipedia](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard)
7. RSA (Rivest–Shamir–Adleman): [RSA (cryptosystem) - Wikipedia](https://en.wikipedia.org/wiki/RSA_(cryptosystem))
8. ECC (Elliptic Curve Cryptography): [Elliptic curve cryptography - Wikipedia](https://en.wikipedia.org/wiki/Elliptic_curve_cryptography)
9. SHA (Secure Hash Algorithm): [Secure Hash Algorithm - Wikipedia](https://en.wikipedia.org/wiki/Secure_Hash_Algorithm)

# 2. Authentication

# 3. Key, Random Number and Advanced Technology

