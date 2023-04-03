from pydantic import BaseModel, Field

PAY_0_SCALE = """ Repayment status in September, 2005 (-2= no credit to pay,-1=pay duly,0= minimum payment is met,
 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay 
 for nine months and above) ."""


class CreditData(BaseModel):
    LIMIT_BAL: int = Field(title=" Amount of the given credit (NT dollar)")
    SEX: int = Field(title="1=male, 2=female")
    EDUCATION: int = Field(title="1 = graduate school; 2 = university; 3 = high school; 4 = others",)
    MARRIAGE: int = Field(title="1 = married; 2 = single; 3 = others")
    AGE: int = Field(title="Age in years .", gt=19, lt=100)
    PAY_0: int =  Field(title=PAY_0_SCALE, lt=10)
    PAY_2: int = Field(title="Repayment status in August 2005 (scale same as above)", lt=10)
    PAY_3: int = Field(title="Repayment status in July 2005 (scale same as above)", lt=10)
    PAY_4: int= Field(title="Repayment status in June 2005 (scale same as above)", lt=10)
    PAY_5: int= Field(title="Repayment status in May 2005 (scale same as above)", lt=10)
    PAY_6: int= Field(title="Repayment status in April 2005 (scale same as above)", lt=10)
    BILL_AMT1: int = Field(title="Amount of bill statement in September 2005 (NT dollar)")
    BILL_AMT2: int = Field(title="Amount of bill statement in August 2005 (NT dollar) .")
    BILL_AMT3: int = Field(title="Amount of bill statement in July 2005 (NT dollar)")
    BILL_AMT4: int = Field(title="Amount of bill statement in June 2005 (NT dollar)")
    BILL_AMT5: int = Field(title="Amount of bill statement in May 2005 (NT dollar)")
    BILL_AMT6: int = Field(title="Amount of bill statement in April 2005 (NT dollar)")
    PAY_AMT1: int = Field(title="Amount of previous payment in September 2005 (NT dollar) .")
    PAY_AMT2: int = Field(title="Amount of previous payment in August 2005 (NT dollar) .")
    PAY_AMT3: int = Field(title="Amount of previous payment in July 2005 (NT dollar) .")
    PAY_AMT4: int = Field(title="Amount of previous payment in June 2005 (NT dollar) .")
    PAY_AMT5: int = Field(title="Amount of previous payment in May 2005 (NT dollar) .")
    PAY_AMT6: int = Field(title="Amount of previous payment in April 2005 (NT dollar) .")

   