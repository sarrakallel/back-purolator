from django.db import models

# Create your models here.


class Employee(models.Model):
    id = models.AutoField(primary_key=True)
    firstname = models.CharField(max_length=100)
    lastname = models.CharField(max_length=100)
    password = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.firstname} {self.lastname}"


class Consignee(models.Model):
    ConsigneeID = models.IntegerField(primary_key=True)
    Name = models.CharField(max_length=100)
    Stackable = models.BooleanField()

    def __str__(self):
        return self.Name


class CPCBinLoading(models.Model):
    CPCLoadingID = models.AutoField(primary_key=True)
    CPCBin_ID = models.IntegerField()
    Date_of_Load = models.DateTimeField()
    Finish_of_Load = models.DateTimeField()
    Stackable = models.BooleanField()
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    consignee = models.ForeignKey(Consignee, on_delete=models.CASCADE)

    def __str__(self):
        return f"CPCLoadingID: {self.CPCLoadingID}, CPCBin_ID: {self.CPCBin_ID}, Date_of_Load: {self.Date_of_Load}, Finish_of_Load: {self.Finish_of_Load}, Stackable: {self.Stackable}"

class PalletLoading(models.Model):
    PalletLoadingID = models.AutoField(primary_key=True)
    Pallet_ID = models.IntegerField()
    Date_of_Load = models.DateTimeField()
    Finish_of_Load = models.DateTimeField()
    Stackable = models.BooleanField()
    Type = models.CharField(max_length=100)

    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    consignee = models.ForeignKey(Consignee, on_delete=models.CASCADE)

    def __str__(self):
        return f"PalletLoadingID: {self.PalletLoadingID}, Pallet_ID: {self.Pallet_ID}"
    
    