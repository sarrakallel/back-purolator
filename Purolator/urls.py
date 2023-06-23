from django.contrib import admin
from django.urls import path
from Application.views import EmployeeLoginView
from rest_framework_swagger.views import get_swagger_view
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework import permissions
from Application.views import MeasurementView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('employee-login/', EmployeeLoginView.as_view(), name='employee-login'),
    path('api/measurement/', MeasurementView.as_view(), name='measurement-api'),

]

schema_view = get_swagger_view(title='API Documentation')

urlpatterns += [
    path('swagger/', schema_view),
]

schema_view = get_schema_view(
    openapi.Info(
        title="Your API",
        default_version="v1",
        description="API documentation",
        terms_of_service="https://www.example.com/terms/",
        contact=openapi.Contact(email="contact@example.com"),
        license=openapi.License(name="MIT License"),
    ),
    public=True,
    permission_classes=[permissions.AllowAny],
)

urlpatterns += [
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]
