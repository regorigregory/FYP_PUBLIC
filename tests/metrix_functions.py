import numpy as np
import unittest
import logging
from components.utils.Metrix import Wrapper as m
logger = logging.getLogger('padding_addition')
logger.setLevel(logging.DEBUG)

class TestMetrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._metrixInstance = m
        cls._testFunctions = list([m.bad, m.avgerr, m.eucledian_distance, m.mse])
        cls._testParameters = dict()
        cls._testParameters["identity"] = [TestMetrix.getRandomIdentityArrays()]
        cls._testParameters["identityAndOcclusions"] = [TestMetrix.getRandomIdentityArraysWithOccMap()]
        cls._testParameters["identityAndOcclusions2"] = [TestMetrix.getRandomIdentityArraysWithOccMap2()]

    @staticmethod
    def getRandomIdentityArrays():
        rand_array = np.random.randint(0, 10000, [300,300])
        copy_of_rand_array = rand_array.copy()
        return rand_array, copy_of_rand_array, None

    @staticmethod
    def getRandomIdentityArraysWithOccMap():
        rand_array = np.empty([300,300])
        rand_array.fill(3)

        occlusions = np.ones([300,300])
        occlusions[0, :] = 0
        copy_of_rand_array = rand_array.copy()
        return rand_array, copy_of_rand_array, occlusions

    @staticmethod
    def getRandomIdentityArraysWithOccMap2():
        rand_array = np.empty([300, 300])
        rand_array.fill(3)

        occlusions = np.ones([300, 300])
        occlusions[0, :] = 0
        copy_of_rand_array = rand_array.copy()
        copy_of_rand_array[0, :] =0
        return rand_array, copy_of_rand_array, occlusions

    def testIdentity(self):
        disp, gt, _ = TestMetrix._testParameters["identity"][0]
        for f in TestMetrix._testFunctions:
            result = f(disp, gt, None)
            failure_message = "\nThere has been an assertion error when testing the following function: {0}".format(f)
            self.assertEqual(result, 0, failure_message)

    def testIfOcclusionsAreNotErrors1(self):
        disp, gt, occ = TestMetrix._testParameters["identityAndOcclusions"][0]
        for f in TestMetrix._testFunctions:
            result = f(disp, gt, occ, occlusions_counted_in_errors=False)
            failure_message = "\nThere has been an assertion error when testing the following function: {0}".format(f)
            self.assertEqual(result, 0, failure_message)

    def testIfOcclusionsAreNotErrors2(self):
        disp, gt, occ = TestMetrix._testParameters["identityAndOcclusions"][0]
        disp[0, :] = 100
        for f in TestMetrix._testFunctions:
            result = f(disp, gt, occ, occlusions_counted_in_errors=False)
            failure_message = "\nThere has been an assertion error when testing the following function: {0}".format(f)
            self.assertEqual(result, 0, failure_message)

    def testBadThreshold(self):
        disp, gt, occ = TestMetrix._testParameters["identityAndOcclusions2"][0]
        result = TestMetrix._metrixInstance.bad(disp, gt, occ, occlusions_counted_in_errors=True, threshold= 3)


    def testIfOcclusionsAreErrors(self):
        disp, gt, occ = TestMetrix._testParameters["identityAndOcclusions2"][0]
        num_pixels = disp.size
        occluded_pixels = np.sum(occ==0)
        difference = disp[0, :] - gt[0, :]
        assert occluded_pixels==300
        assert np.sum(difference) == 300*3, "{0} is not 900".format(np.sum(difference))
        #        cls._testFunctions = list([m.bad, m.avgerr, m.eucledian_distance, m.mse])
        expected_results = []
        expected_results.append(300/num_pixels)
        expected_results.append(np.sum(np.abs(difference))/num_pixels)
        expected_results.append(np.sqrt(np.sum(np.power(disp-gt, 2))))
        expected_results.append(np.sum(np.power(difference, 2)) / num_pixels)
        for i, f in enumerate(TestMetrix._testFunctions):
            result = f(disp, gt, occ, occlusions_counted_in_errors=True)
            failure_message = "\nThere has been an assertion error when testing the following function: {0}".format(f)
            self.assertEqual(result, expected_results[i], failure_message)


if __name__ == "__main__":
    unittest.main()

