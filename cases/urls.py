from django.urls import path
from . import views

app_name = "cases"

urlpatterns = [
    path("", views.home, name="home"),
    path("case/<int:case_id>/", views.case_detail, name="case_detail"),
    path("case/<int:case_id>/related/", views.related_cases, name="related_cases"),
    path("dashboard/", views.dashboard, name="dashboard"),
    
    # Authentication Routes
    path("admin-login/", views.admin_login, name="admin_login"),
    path("admin-logout/", views.admin_logout, name="admin_logout"),

    # Admin Console Routes (Protected - Staff / Superuser Only)
    path("admin-console/", views.admin_dashboard, name="admin_dashboard"),
    path("admin-console/research/", views.research_console, name="research_console"),
    path("admin-console/corpus/", views.admin_corpus, name="admin_corpus"),
    path("admin-console/add-case/", views.admin_add_case, name="admin_add_case"),
    path("admin-console/upload-judgment/", views.admin_add_case, name="admin_upload_judgment"),
    path("admin-console/preview-judgment/", views.admin_preview_judgment, name="admin_preview_judgment"),
    path("admin-console/confirm-ingestion/", views.admin_confirm_ingestion, name="admin_confirm_ingestion"),
    path("admin-console/rebuild-index/", views.admin_rebuild_index, name="admin_rebuild_index"),
    path("admin-console/ingestion-history/", views.admin_ingestion_history, name="admin_ingestion_history"),
    
    # Legacy alias redirect/compatibility route for research console
    path("research-console/", views.research_console, name="research_console_legacy"),
    
    # Legal Workstation & Research Tools
    path("compare-cases/", views.case_compare, name="case_compare"),
    path("compare/", views.case_compare, name="case_compare_legacy"),
    path("export-report/", views.export_report, name="export_report"),
    
    path("evaluation/", views.evaluation, name="evaluation"),
    path("saved-cases/", views.saved_cases, name="saved_cases"),
    path("save-case/<int:case_id>/", views.toggle_save_case, name="toggle_save_case"),
    path("assistant/", views.plain_language_assistant, name="assistant"),
    path("tamil-nadu-legal-help/", views.lawyer_finder, name="lawyer_finder"),
    path("legal-awareness/", views.legal_awareness, name="legal_awareness"),
    path("about/", views.about, name="about"),
    path("annotate/", views.annotate_benchmark, name="annotate_benchmark"),
    path("annotate/save/", views.save_annotation, name="save_annotation"),
    path("research-status/", views.research_status, name="research_status"),
]
