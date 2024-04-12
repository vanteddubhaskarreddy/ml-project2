import sys
import os
from src.logger import logging

def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error in \n file {file_name} at \n line {exc_tb.tb_lineno} \n error message: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys) -> None:
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
    
    def __str__(self) -> str:
        return f"Error: {self.error_message}"
    

if __name__ == '__main__':
    try:
        1/0
    except Exception as e:
        logging.info('Division by zero error')
        raise CustomException(e, sys)