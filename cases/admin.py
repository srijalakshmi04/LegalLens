from django.contrib import admin

from .models import LegalCase, ModelTrainingRun, SearchHistory


@admin.register(LegalCase)
class LegalCaseAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "judgment_year",
        "judgment_date",
        "author",
        "page_count",
        "imported_at",
    )

    list_filter = (
        "judgment_year",
        "judgment_date",
        "imported_at",
    )

    search_fields = (
        "title",
        "citations",
        "author",
        "petitioner",
        "respondent",
    )

    readonly_fields = (
        "file_checksum",
        "imported_at",
        "updated_at",
    )

    ordering = ("-judgment_date",)


@admin.register(SearchHistory)
class SearchHistoryAdmin(admin.ModelAdmin):
    list_display = (
        "short_query",
        "results_count",
        "top_similarity_score",
        "user",
        "searched_at",
    )

    list_filter = ("searched_at",)
    search_fields = ("query_text",)

    @admin.display(description="Query")
    def short_query(self, obj):
        if len(obj.query_text) <= 70:
            return obj.query_text

        return f"{obj.query_text[:70]}..."


@admin.register(ModelTrainingRun)
class ModelTrainingRunAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "status",
        "case_count",
        "feature_count",
        "started_at",
        "completed_at",
    )

    list_filter = ("status",)
    readonly_fields = (
        "started_at",
        "completed_at",
    )