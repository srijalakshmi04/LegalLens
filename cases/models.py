from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class LegalCase(models.Model):
    class ExtractionStatus(models.TextChoices):
        UNREVIEWED = "unreviewed", "Unreviewed"
        COMPLETE = "complete", "Complete"
        PARTIAL = "partial", "Partial"
        SCANNED = "scanned", "Scanned PDF"
        FAILED = "failed", "Failed"

    title = models.CharField(max_length=500, db_index=True)
    filename = models.CharField(max_length=500, unique=True)
    file_path = models.TextField()

    judgment_year = models.PositiveSmallIntegerField(db_index=True)
    judgment_date = models.DateField(
        null=True,
        blank=True,
        db_index=True,
    )

    citations = models.TextField(blank=True)
    author = models.CharField(max_length=300, blank=True)
    bench = models.TextField(blank=True)
    petitioner = models.CharField(max_length=500, blank=True)
    respondent = models.CharField(max_length=500, blank=True)
    acts = models.TextField(blank=True)
    legal_sections = models.TextField(blank=True)

    raw_text = models.TextField()
    cleaned_text = models.TextField()

    page_count = models.PositiveIntegerField(default=0)
    word_count = models.PositiveIntegerField(default=0)

    extraction_status = models.CharField(
        max_length=20,
        choices=ExtractionStatus.choices,
        default=ExtractionStatus.UNREVIEWED,
        db_index=True,
    )

    extraction_quality = models.FloatField(
        default=0.0,
        validators=[
            MinValueValidator(0.0),
            MaxValueValidator(1.0),
        ],
    )

    requires_ocr = models.BooleanField(
        default=False,
        db_index=True,
    )

    file_checksum = models.CharField(
        max_length=64,
        unique=True,
        null=True,
        blank=True,
    )

    imported_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = [
            "-judgment_date",
            "-judgment_year",
            "title",
        ]
        indexes = [
            models.Index(
                fields=["judgment_year", "title"],
                name="case_year_title_idx",
            ),
            models.Index(
                fields=["extraction_status", "requires_ocr"],
                name="case_extract_ocr_idx",
            ),
        ]

    def __str__(self):
        return f"{self.title} ({self.judgment_year})"


class SearchHistory(models.Model):
    """
    Store search statistics and privacy-safe display summaries without storing sensitive raw inputs.
    """

    query_fingerprint = models.CharField(
        max_length=64,
        db_index=True,
    )

    sanitized_query = models.CharField(
        max_length=150,
        blank=True,
        default="",
        help_text="Privacy-safe truncated display summary of query.",
    )

    search_mode = models.CharField(
        max_length=20,
        default="all",
        blank=True,
    )

    query_word_count = models.PositiveIntegerField(default=0)
    results_count = models.PositiveIntegerField(default=0)

    top_similarity_score = models.FloatField(
        null=True,
        blank=True,
    )

    processing_time_ms = models.PositiveIntegerField(default=0)

    model_version = models.CharField(
        max_length=50,
        blank=True,
    )

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="legal_case_searches",
    )

    searched_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-searched_at"]
        verbose_name_plural = "Search histories"

    @property
    def display_query(self):
        if self.sanitized_query and self.sanitized_query.strip():
            return self.sanitized_query
        return f"Legal Search ({self.query_word_count} words)"

    def __str__(self):
        return (
            f"Search '{self.display_query[:30]}' "
            f"on {self.searched_at:%d-%m-%Y %H:%M}"
        )


class ModelTrainingRun(models.Model):
    class Status(models.TextChoices):
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.RUNNING,
    )

    model_version = models.CharField(
        max_length=50,
        blank=True,
    )

    algorithm = models.CharField(
        max_length=200,
        default="TF-IDF with cosine similarity",
    )

    case_count = models.PositiveIntegerField(default=0)
    feature_count = models.PositiveIntegerField(default=0)

    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(
        null=True,
        blank=True,
    )

    error_message = models.TextField(blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self):
        return f"Training {self.id}: {self.status}"


class SavedCase(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="saved_cases",
    )
    session_key = models.CharField(
        max_length=40,
        blank=True,
        db_index=True,
    )
    legal_case = models.ForeignKey(
        LegalCase,
        on_delete=models.CASCADE,
        related_name="saved_by_users",
    )
    notes = models.TextField(blank=True)
    saved_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-saved_at"]
        verbose_name_plural = "Saved cases"

    def __str__(self):
        return f"Saved case: {self.legal_case.title}"