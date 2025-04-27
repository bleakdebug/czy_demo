import sys
print("Python path:", sys.path)

try:
    import unKR
    print("unKR package location:", unKR.__file__)
    
    from unKR.data.SAURData import SAURDataModule
    print("Successfully imported SAURDataModule")
    
except Exception as e:
    print("Error:", str(e))
    import traceback
    traceback.print_exc() 