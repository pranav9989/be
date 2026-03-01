"""
email_service.py – Sends transactional emails via Gmail SMTP.
Uses App Password (not regular Gmail password) stored in .env
"""

import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

# ─── Frontend base URL (change for production) ───────────────────────────────
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")


def send_password_reset_email(to_email: str, reset_token: str) -> bool:
    """
    Send a branded HTML password-reset email.
    Returns True on success, False on failure.
    """
    reset_link = f"{FRONTEND_URL}/reset-password?token={reset_token}"

    subject = "Reset your CS Interview Assistant password"

    html_body = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Reset Password</title>
</head>
<body style="margin:0;padding:0;background:#0F0F11;font-family:'Segoe UI',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0F0F11;padding:40px 16px;">
    <tr>
      <td align="center">
        <table width="560" cellpadding="0" cellspacing="0"
               style="background:#18181C;border:1px solid #2A2A35;border-radius:20px;overflow:hidden;max-width:560px;width:100%;">

          <!-- Header -->
          <tr>
            <td style="background:linear-gradient(135deg,#1A1035 0%,#0F0F18 100%);padding:36px 40px 28px;text-align:center;">
              <table cellpadding="0" cellspacing="0" style="margin:0 auto 20px;">
                <tr>
                  <td style="background:linear-gradient(135deg,#D4A853,#B8860B);width:56px;height:56px;border-radius:14px;text-align:center;vertical-align:middle;">
                    <span style="font-size:26px;line-height:56px;">🧠</span>
                  </td>
                </tr>
              </table>
              <h1 style="margin:0 0 6px;font-size:22px;font-weight:700;color:#F2F2F5;letter-spacing:-0.5px;">
                CS Interview Assistant
              </h1>
              <p style="margin:0;font-size:13px;color:#8A8A9A;">Your AI-powered interview preparation companion</p>
            </td>
          </tr>

          <!-- Body -->
          <tr>
            <td style="padding:36px 40px;">
              <h2 style="margin:0 0 12px;font-size:20px;font-weight:700;color:#F2F2F5;">
                Reset your password 🔑
              </h2>
              <p style="margin:0 0 24px;font-size:15px;line-height:1.65;color:#8A8A9A;">
                We received a request to reset the password for your account.
                Click the button below to create a new password.
                This link is valid for <strong style="color:#D4A853;">15 minutes</strong>.
              </p>

              <!-- CTA Button -->
              <table cellpadding="0" cellspacing="0" style="margin:0 0 28px;">
                <tr>
                  <td style="background:linear-gradient(135deg,#D4A853,#B8860B);border-radius:12px;">
                    <a href="{reset_link}"
                       style="display:inline-block;padding:14px 36px;font-size:15px;font-weight:700;
                              color:#0F0F0F;text-decoration:none;letter-spacing:0.01em;">
                      Reset My Password →
                    </a>
                  </td>
                </tr>
              </table>

              <!-- Fallback link -->
              <p style="margin:0 0 8px;font-size:13px;color:#6A6A7A;">
                If the button doesn't work, copy and paste this link:
              </p>
              <p style="margin:0 0 28px;word-break:break-all;">
                <a href="{reset_link}" style="font-size:12px;color:#D4A853;text-decoration:none;">{reset_link}</a>
              </p>

              <!-- Warning box -->
              <table cellpadding="0" cellspacing="0" width="100%"
                     style="background:#1E1015;border:1px solid rgba(239,68,68,0.2);border-radius:10px;margin-bottom:24px;">
                <tr>
                  <td style="padding:14px 18px;">
                    <p style="margin:0;font-size:13px;color:#F87171;line-height:1.6;">
                      ⚠️ If you did not request this password reset, you can safely ignore this email.
                      Your password will remain unchanged.
                    </p>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="padding:20px 40px 28px;border-top:1px solid #2A2A35;text-align:center;">
              <p style="margin:0;font-size:12px;color:#4A4A5A;line-height:1.6;">
                This email was sent by CS Interview Assistant.<br />
                If you have questions, contact your administrator.
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""

    plain_body = (
        f"CS Interview Assistant – Password Reset\n\n"
        f"Click the link below to reset your password (valid for 15 minutes):\n"
        f"{reset_link}\n\n"
        f"If you did not request this, you can safely ignore this email."
    )

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"CS Interview Assistant <{GMAIL_USER}>"
        msg["To"] = to_email

        msg.attach(MIMEText(plain_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_USER, to_email, msg.as_string())

        print(f"[EmailService] Reset email sent to {to_email}")
        return True

    except smtplib.SMTPAuthenticationError:
        print("[EmailService] ERROR: Gmail authentication failed. Check GMAIL_USER and GMAIL_APP_PASSWORD in .env")
        return False
    except Exception as e:
        print(f"[EmailService] ERROR sending email: {e}")
        return False
