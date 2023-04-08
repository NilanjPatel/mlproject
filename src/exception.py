import sys
import logger
import logging

def error_message_details(error, error_details: sys):
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number[{1}] error message".format(
        file_name, exc_tb.tb_lineno, str(error)

    )

    return error_message


class CustomException(Exception):
    def __int__(self, error_message, error_details: sys):
        super().__init__(error_message)
        print('eoor msg', error_message)
        self.error_message = error_message_details(error_message, error_details == error_details)


    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.error("Divide by Zero Error")
        print('eror')
        raise CustomException(e, sys)