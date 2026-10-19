from django.conf import settings


def admin_status(request):
    """Context processor providing is_admin flag for template navigation."""
    user = getattr(request, "user", None)
    is_admin = bool(
        user
        and user.is_authenticated
        and (user.is_staff or user.is_superuser)
    )
    return {"is_admin": is_admin}
