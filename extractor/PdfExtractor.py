import os

class PdfExtractor:
    @staticmethod
    def pdf_to_imgs(pdf_file_name):
        command = (
                "convert -background white  -alpha remove -alpha off -density 200 '"
                + pdf_file_name
                + "'[0-12]  png24:"
                + os.path.splitext(pdf_file_name)[0]
                + "-%04d.png"
        )
        return os.system(command)
