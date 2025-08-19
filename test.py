import unittest
import model
from model import *



class TestGetModel(unittest.TestCase):
    def test_valid_input(self):
        # Test case 1: Valid car model with matching model code
        text1 = "I want to order an ix xDrive50 with sunroof."
        self.assertEqual(getmodel(text1), '21CF')

        # Test case 2: Valid car model with matching model code (different capitalization)
        text2 = "I want to order an iX xdrive40."
        self.assertEqual(getmodel(text2), '11CF')

    def test_invalid_input(self):
        # Test case 3: Valid car model with no matching model code
        text3 = "I want to order an x7 xDrive35i."
        self.assertIsNone(getmodel(text3))

        # Test case 4: Invalid car model
        text4 = "I want to order a 320i."
        self.assertIsNone(getmodel(text4))

class TestGetDate(unittest.TestCase):
    def test_numerical_date(self):
        # Test case 1: Valid prompt with numerical date
        text1 = "Please schedule the delivery for 10th June 2023."
        self.assertEqual(getdate(text1), '2023-06-10')


    def test_textual_date(self):
        # Test case 3: Valid prompt with textual date (start of the month)
        text3 = "I want the delivery to be at the start of July 2023."
        self.assertEqual(getdate(text3), '2023-07-01')

        # Test case 4: Valid prompt with textual date (end of the month)
        text4 = "Please schedule the delivery for the end of December 2023."
        self.assertEqual(getdate(text4), '2023-12-30')

        # Test case 5: Valid prompt with textual date (mid of the month)
        text5 = "The delivery should be in mid of March 2023."
        self.assertEqual(getdate(text5), '2023-03-15')

    def test_invalid_input(self):
        # Test case 6: Invalid prompt with no valid date
        text6 = "Please schedule the delivery as soon as possible."
        self.assertIsNone(getdate(text6))



class TestGetConfigurations(unittest.TestCase):
    def test_valid_input(self):
        # Test case 1: Valid prompt with multiple configurations
        text1 = "The car should have left-hand drive and m sport package along with sunroof."
        expected_result1 = ['LL','+ ','P337A','+', 'S403A']
        self.assertEqual(get_configurations(text1), expected_result1)

        # Test case 2: Valid prompt with configurations and connecting words
        text2 = "The car should have right-hand drive or panorama glass roof and without m sport package."
        expected_result2 = ['RL', '/', 'S402A', '-','P337A']   #precedence is done during final print
        self.assertEqual(get_configurations(text2), expected_result2)

    def test_invalid_input(self):
        # Test case 3: Invalid prompt with no configurations
        text3 = "Please provide the car specifications."
        self.assertEqual(get_configurations(text3), [])

        # Test case 4: Invalid prompt with invalid configuration
        text4 = "The car should have xyz package and sunroof."
        self.assertEqual(get_configurations(text4), ['+','sunroof'])


    def test_empty_input(self):
        # Test case 6: Empty prompt
        text6 = ""
        self.assertEqual(get_configurations(text6), [])

    def test_case_sensitivity(self):
        # Test case 7: Valid prompt with case-sensitive configurations
        text7 = "The car with Panorama Glass Roof and Sunroof."
        expected_result7 = ['+','S402A','+' 'S403A']
        self.assertEqual(get_configurations(text7), expected_result7)



if __name__ == '__main__':
    unittest.main()


