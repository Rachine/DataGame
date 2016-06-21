import unittest
from img_processing import Image, walk_directory


class TestImgProcessing(unittest.TestCase):

    def setUp(self):
        super(TestImgProcessing, self).setUp()
        self.filename = "-1191173.jpg"

    def test_img_loader(self):
        i = Image(self.filename)
        self.assertTrue(i.img is not None)
        self.assertEqual(len(i.img.shape), 3)

    def test_directory_discovery(self):
        # Check if all the images are found in the directory
        img_files = walk_directory()
        self.assertTrue(len(img_files) > 0)
        self.assertEqual(len(img_files), 5360)


if __name__ == '__main__':
    unittest.main()
