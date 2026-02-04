"""
Security Audit Logger
Provides structured logging for security-relevant events.
Follows OWASP logging best practices.
"""

from datetime import datetime
from typing import Any
from loguru import logger


class AuditLogger:
    """
    Centralized security audit logger.
    Logs security events without exposing sensitive data.
    """

    @staticmethod
    def log_connection(client_uid: str, client_ip: str, event: str) -> None:
        """
        Log WebSocket connection events.

        Args:
            client_uid: Client unique identifier
            client_ip: Client IP address (may be masked for privacy)
            event: Event type (connected, disconnected, rate_limited)
        """
        # Mask last octet of IP for privacy in logs
        masked_ip = AuditLogger._mask_ip(client_ip)
        logger.info(
            f"[AUDIT] Connection {event}: client={client_uid[:8]}... ip={masked_ip}"
        )

    @staticmethod
    def log_rate_limit(
        client_uid: str, client_ip: str, limit_type: str, count: int
    ) -> None:
        """
        Log rate limit violations.

        Args:
            client_uid: Client unique identifier
            client_ip: Client IP address
            limit_type: Type of rate limit (connection, message, upload)
            count: Number of requests that triggered the limit
        """
        masked_ip = AuditLogger._mask_ip(client_ip)
        logger.warning(
            f"[AUDIT] Rate limit exceeded: type={limit_type} "
            f"client={client_uid[:8] if client_uid else 'unknown'}... "
            f"ip={masked_ip} count={count}"
        )

    @staticmethod
    def log_validation_failure(client_uid: str, message_type: str, reason: str) -> None:
        """
        Log input validation failures.

        Args:
            client_uid: Client unique identifier
            message_type: Type of message that failed validation
            reason: Brief reason for failure (no sensitive data)
        """
        logger.warning(
            f"[AUDIT] Validation failed: client={client_uid[:8]}... "
            f"type={message_type} reason={reason}"
        )

    @staticmethod
    def log_config_change(client_uid: str, config_name: str) -> None:
        """
        Log configuration changes.

        Args:
            client_uid: Client unique identifier
            config_name: Name of configuration that was changed to
        """
        logger.info(
            f"[AUDIT] Config change: client={client_uid[:8]}... config={config_name}"
        )

    @staticmethod
    def log_file_access(
        client_uid: str, file_type: str, action: str, success: bool
    ) -> None:
        """
        Log file access events (uploads, history access).

        Args:
            client_uid: Client unique identifier
            file_type: Type of file (audio, history, config)
            action: Action performed (upload, read, delete)
            success: Whether the action succeeded
        """
        status = "success" if success else "failed"
        logger.info(
            f"[AUDIT] File access: client={client_uid[:8]}... "
            f"type={file_type} action={action} status={status}"
        )

    @staticmethod
    def _mask_ip(ip: str) -> str:
        """
        Mask IP address for privacy in logs.
        Keeps first 3 octets for IPv4, masks last octet.

        Args:
            ip: IP address to mask

        Returns:
            Masked IP address
        """
        if not ip or ip == "unknown":
            return "unknown"
        parts = ip.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.{parts[2]}.xxx"
        # For IPv6 or other formats, just truncate
        return ip[:10] + "..."


# Global audit logger instance
audit_logger = AuditLogger()
