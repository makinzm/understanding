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

It is a form of security through obscurity, where the existence of the message is hidden rather than just its content.

# 2. Authentication

# 3. Key, Random Number and Advanced Technology

