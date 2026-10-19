from django.test import TestCase
from django.urls import reverse
from cases.forms import LawyerFinderForm
from cases.services.lawyer_directory import find_tamil_nadu_help, get_directory_stats

class TamilNaduLegalHelpDirectoryTest(TestCase):
    ALL_38_DISTRICTS = [
        "Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Cuddalore",
        "Dharmapuri", "Dindigul", "Erode", "Kallakurichi", "Kancheepuram",
        "Kanyakumari", "Karur", "Krishnagiri", "Madurai", "Mayiladuthurai",
        "Nagapattinam", "Namakkal", "Perambalur", "Pudukkottai", "Ramanathapuram",
        "Ranipet", "Salem", "Sivaganga", "Tenkasi", "Thanjavur",
        "Theni", "The Nilgiris", "Thiruvallur", "Thiruvarur", "Thoothukudi",
        "Tiruchirappalli", "Tirunelveli", "Tirupathur", "Tiruppur", "Tiruvannamalai",
        "Vellore", "Viluppuram", "Virudhunagar"
    ]

    def test_all_38_districts_in_form_choices(self):
        form_choices = [choice[0] for choice in LawyerFinderForm.DISTRICTS if choice[0]]
        self.assertEqual(len(form_choices), 38)
        for d in self.ALL_38_DISTRICTS:
            self.assertIn(d, form_choices)

    def test_find_tamil_nadu_help_for_all_38_districts(self):
        for d in self.ALL_38_DISTRICTS:
            res = find_tamil_nadu_help(d)
            self.assertIsNotNone(res["district_info"], f"District info missing for {d}")
            self.assertEqual(res["district_info"]["district"].casefold(), d.casefold())
            self.assertIn(".nic.in", res["district_info"]["official_district_website"])
            self.assertTrue(len(res["offices"]) > 0, f"DLSA missing for {d}")
            self.assertTrue(len(res["courts"]) > 0, f"eCourts missing for {d}")
            self.assertIn("nalsa.gov.in", res["offices"][0]["verification_url"].lower())
            self.assertTrue("ecourts.gov.in" in res["courts"][0]["official_ecourts_url"].lower() or "dcourts.gov.in" in res["courts"][0]["official_ecourts_url"].lower())

    def test_lawyer_finder_view_status_200_for_all_districts(self):
        url = reverse("cases:lawyer_finder")
        for d in self.ALL_38_DISTRICTS:
            response = self.client.get(url, {"district": d})
            self.assertEqual(response.status_code, 200, f"HTTP status not 200 for district {d}")
            self.assertContains(response, d)
            self.assertNotContains(response, "No local district entry bundled")
            self.assertNotContains(response, "No court entry bundled")

    def test_directory_coverage_stats(self):
        stats = get_directory_stats()
        self.assertEqual(stats["configured_districts"], 38)
        self.assertEqual(stats["total_districts"], 38)
        self.assertEqual(stats["district_websites_count"], 38)
        self.assertEqual(stats["dlsa_count"], 38)
        self.assertEqual(stats["ecourts_count"], 38)

    def test_district_and_category_filtering(self):
        # 1. Coimbatore + Criminal
        res1 = find_tamil_nadu_help("Coimbatore", specialization="criminal")
        self.assertEqual(res1["district"], "Coimbatore")
        self.assertEqual(res1["district_info"]["district"], "Coimbatore")
        self.assertTrue(len(res1["offices"]) > 0)

        # 2. Madurai
        res2 = find_tamil_nadu_help("Madurai")
        self.assertEqual(res2["district"], "Madurai")
        self.assertEqual(res2["district_info"]["district"], "Madurai")
        self.assertTrue(len(res2["offices"]) > 0)

        # 3. View GET request for Coimbatore
        url = reverse("cases:lawyer_finder")
        response = self.client.get(url, {"district": "Coimbatore", "specialization": "criminal"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Coimbatore")
        self.assertNotContains(response, "PIN")
        self.assertNotContains(response, "District Mismatch")
