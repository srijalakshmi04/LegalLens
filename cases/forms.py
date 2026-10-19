from django import forms


class CaseSearchForm(forms.Form):
    case_description = forms.CharField(
        label="Legal Issue, Fact Pattern, or Provision",
        min_length=20,
        max_length=5000,
        widget=forms.Textarea(
            attrs={
                "rows": 8,
                "placeholder": (
                    "Describe the facts, legal issue, applicable "
                    "law sections, and relief requested..."
                ),
                "class": "case-search-input",
            }
        ),
    )

    result_count = forms.IntegerField(
        required=False,
        initial=5,
        min_value=1,
        max_value=20,
        widget=forms.HiddenInput(),
    )

    # Legacy parameters gracefully accepted if submitted by old links or tests, but never required
    search_mode = forms.CharField(required=False, widget=forms.HiddenInput())
    start_year = forms.IntegerField(required=False, widget=forms.HiddenInput())
    end_year = forms.IntegerField(required=False, widget=forms.HiddenInput())
    legal_provision = forms.CharField(required=False, widget=forms.HiddenInput())
    court_bench = forms.CharField(required=False, widget=forms.HiddenInput())

    def clean_case_description(self):
        description = self.cleaned_data[
            "case_description"
        ]

        return " ".join(description.split())


class PlainLanguageAssistantForm(forms.Form):
    issue = forms.CharField(
        label="Tell us what happened",
        min_length=20,
        max_length=3000,
        widget=forms.Textarea(
            attrs={
                "rows": 7,
                "placeholder": (
                    "Explain the problem in your own words. "
                    "For example: My employer removed me from my job "
                    "without giving a reason or notice."
                ),
                "class": "case-search-input",
            }
        ),
    )

    def clean_issue(self):
        return " ".join(self.cleaned_data["issue"].split())


class LawyerFinderForm(forms.Form):
    DISTRICTS = (
        ("", "Select a Tamil Nadu district (All 38 Districts)"),
        ("Ariyalur", "Ariyalur"),
        ("Chengalpattu", "Chengalpattu"),
        ("Chennai", "Chennai"),
        ("Coimbatore", "Coimbatore"),
        ("Cuddalore", "Cuddalore"),
        ("Dharmapuri", "Dharmapuri"),
        ("Dindigul", "Dindigul"),
        ("Erode", "Erode"),
        ("Kallakurichi", "Kallakurichi"),
        ("Kancheepuram", "Kancheepuram"),
        ("Kanyakumari", "Kanyakumari"),
        ("Karur", "Karur"),
        ("Krishnagiri", "Krishnagiri"),
        ("Madurai", "Madurai"),
        ("Mayiladuthurai", "Mayiladuthurai"),
        ("Nagapattinam", "Nagapattinam"),
        ("Namakkal", "Namakkal"),
        ("Perambalur", "Perambalur"),
        ("Pudukkottai", "Pudukkottai"),
        ("Ramanathapuram", "Ramanathapuram"),
        ("Ranipet", "Ranipet"),
        ("Salem", "Salem"),
        ("Sivaganga", "Sivaganga"),
        ("Tenkasi", "Tenkasi"),
        ("Thanjavur", "Thanjavur"),
        ("Theni", "Theni"),
        ("The Nilgiris", "The Nilgiris"),
        ("Thiruvallur", "Thiruvallur"),
        ("Thiruvarur", "Thiruvarur"),
        ("Thoothukudi", "Thoothukudi"),
        ("Tiruchirappalli", "Tiruchirappalli"),
        ("Tirunelveli", "Tirunelveli"),
        ("Tirupathur", "Tirupathur"),
        ("Tiruppur", "Tiruppur"),
        ("Tiruvannamalai", "Tiruvannamalai"),
        ("Vellore", "Vellore"),
        ("Viluppuram", "Viluppuram"),
        ("Virudhunagar", "Virudhunagar"),
    )
    SPECIALIZATIONS = (
        ("", "Any specialization"),
        ("criminal", "Criminal law"), ("family", "Family law"),
        ("property", "Property and tenancy law"),
        ("consumer", "Consumer law"),
        ("employment", "Employment and labour law"),
        ("cyber", "Cyber law"), ("civil", "Civil law"),
    )

    district = forms.ChoiceField(
        choices=DISTRICTS,
        widget=forms.Select(attrs={"class": "select-input"}),
    )
    specialization = forms.ChoiceField(
        choices=SPECIALIZATIONS,
        required=False,
        widget=forms.Select(attrs={"class": "select-input"}),
    )
