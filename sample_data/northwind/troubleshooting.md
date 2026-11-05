# Troubleshooting

## Emails are not arriving

Check that forwarding from your mail provider to your Northwind Desk forwarding address is active. Gmail and Outlook may require you to confirm forwarding first. Also check the Spam view in Northwind Desk, because messages that fail spam checks are placed there instead of the inbox.

## Replies are going to spam

This usually means the SPF and DKIM records for your domain are missing or wrong. Open Settings > Channels > Email and use the Verify DNS button to see which record is failing.

## I cannot log in

Use the Forgot password link on the login page; the reset link is valid for 1 hour. If your workspace uses SSO, you must log in through your identity provider instead. If you lost access to your two-factor authentication device, ask a workspace admin to reset 2FA for your account.
