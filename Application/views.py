from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Employee
from .models import CPCBinLoading

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Employee
from .camera import MeasurementCamera
from .camera import MeasurementSerializer

class EmployeeLoginView(APIView):
    def post(self, request):
        id = request.data.get('id')
        password = request.data.get('password')
        
        try:
            employee = Employee.objects.get(id=id)
            if employee.password == password:
                # Successful login
                return Response({'message': 'Login successful'}, status=status.HTTP_200_OK)
            else:
                # Invalid password
                return Response({'message': 'Invalid password'}, status=status.HTTP_401_UNAUTHORIZED)
        except Employee.DoesNotExist:
            # Employee not found
            return Response({'message': 'Employee not found'}, status=status.HTTP_404_NOT_FOUND)



class CreateCPCBinView(APIView):
    def post(self, request):
        cpc_bin_id = request.data.get('cpc_bin_id')
        consignee_name = request.data.get('consignee_name')
        stackable = request.data.get('stackable')

        # Create CPCBinLoading object and save to the database
        cpc_bin = CPCBinLoading(cpc_bin_id=cpc_bin_id, consignee_name=consignee_name, stackable=stackable)
        cpc_bin.save()

        return Response({'message': 'CPC Bin created successfully'})


class MeasurementView(APIView):
    def get(self, request):
        # Initialize the MeasurementCamera object
        camera = MeasurementCamera()

        # Perform the measurement and get the results
        results = camera.measure_pallet()

        # Serialize the results
        serializer = MeasurementSerializer({
            'width': results[0],
            'height': results[1],
        })

        return Response(serializer.data)
