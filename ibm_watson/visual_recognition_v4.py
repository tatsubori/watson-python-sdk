# coding: utf-8

# Copyright 2019 IBM All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The IBM Watson&trade; Visual Recognition service uses deep learning algorithms to identify
scenes, objects, and faces  in images you upload to the service. You can create and train
a custom classifier to identify subjects that suit your needs.
"""

from __future__ import absolute_import

import json
from .common import get_sdk_headers
from ibm_cloud_sdk_core import BaseService, DetailedResponse
from ibm_cloud_sdk_core import datetime_to_string, string_to_datetime
from os.path import basename
import re

import requests
from requests.auth import HTTPBasicAuth


def create_annotation(self, label, left, top, width, height):
    return {
        'object': label,
        'location': {
            'left': int(left),
            'top': int(top),
            'width': int(width),
            'height': int(height)
        }
    }


def wrap_response(response, method):
    if 200 <= response.status_code <= 299:
        if response.status_code == 204 or method == 'HEAD':
            # There is no body content for a HEAD request or a 204 response
            return DetailedResponse(None, response.headers, response.status_code)
        if accept_json:
            try:
                response_json = response.json()
            except:
                # deserialization fails because there is no text
                return DetailedResponse(None, response.headers, response.status_code)
            return DetailedResponse(response_json, response.headers, response.status_code)
        return DetailedResponse(response, response.headers, response.status_code)
    else:
        error_message = None
        if response.status_code == 401:
            error_message = 'Unauthorized: Access is denied due to ' \
                            'invalid credentials'
            raise ApiException(response.status_code, error_message, http_response=response)


##############################################################################
# Service
##############################################################################


class VisualRecognitionV4(BaseService):
    """The Visual Recognition V4 service."""

    default_url = 'https://gateway.watsonplatform.net/visual-recognition/api'

    default_version = '2019-02-11'

    def __init__(
            self,
            iam_apikey=None,
            iam_access_token=None,
            iam_url=None,
            iam_client_id=None,
            iam_client_secret=None,
            icp4d_access_token=None,
            icp4d_url=None,
            authentication_type=None,
            version=default_version,
            url=default_url,
            default_headers={},
    ):
        """
        Construct a new client for the Visual Recognition service.

        :param str version: The API version date to use with the service, in
               "YYYY-MM-DD" format. Whenever the API is changed in a backwards
               incompatible way, a new minor version of the API is released.
               The service uses the API version for the date you specify, or
               the most recent version before that date. Note that you should
               not programmatically specify the current date at runtime, in
               case the API has been updated since your application's release.
               Instead, specify a version date that is compatible with your
               application, and don't change it until your application is
               ready for a later version.

        :param str url: The base url to use when contacting the service (e.g.
               "https://gateway.watsonplatform.net/visual-recognition/api/visual-recognition/api").
               The base url may differ between IBM Cloud regions.

        :param str iam_apikey: An API key that can be used to request IAM tokens. If
               this API key is provided, the SDK will manage the token and handle the
               refreshing.

        :param str iam_access_token:  An IAM access token is fully managed by the application.
               Responsibility falls on the application to refresh the token, either before
               it expires or reactively upon receiving a 401 from the service as any requests
               made with an expired token will fail.

        :param str iam_url: An optional URL for the IAM service API. Defaults to
               'https://iam.cloud.ibm.com/identity/token'.

        :param str iam_client_id: An optional client_id value to use when interacting with the IAM service.

        :param str iam_client_secret: An optional client_secret value to use when interacting with the IAM service.

        :param str icp4d_access_token:  A ICP4D(IBM Cloud Pak for Data) access token is
               fully managed by the application. Responsibility falls on the application to
               refresh the token, either before it expires or reactively upon receiving a 401
               from the service as any requests made with an expired token will fail.

        :param str icp4d_url: In order to use an SDK-managed token with ICP4D authentication, this
               URL must be passed in.

        :param str authentication_type: Specifies the authentication pattern to use. Values that it
               takes are basic, iam or icp4d.
        """

        BaseService.__init__(
            self,
            vcap_services_name='watson_vision_combined',
            url=url,
            iam_apikey=iam_apikey,
            iam_access_token=iam_access_token,
            iam_url=iam_url,
            iam_client_id=iam_client_id,
            iam_client_secret=iam_client_secret,
            use_vcap_services=True,
            display_name='Visual Recognition',
            icp4d_access_token=icp4d_access_token,
            icp4d_url=icp4d_url,
            authentication_type=authentication_type)
        self.version = version
        self.default_headers = default_headers
    
    #########################
    # Analysis
    #########################
    
    # https://cloud.ibm.com/apidocs/visual-recognition-v4#analyze-images
    def analyze(
            self,
            collection_ids,
            image_fp,
            threshold=0.5,
            **kwargs):
        """Analyze images by URL, by file, or both against your own
        collection. Make sure that training_status.objects.ready is true
        for the feature before you use a collection to analyze images.
        
        :param float threashold: The minimum score a feature must have to
        be returned.  Constraints: 0.15 <= value <= 1
        
        """
        headers = {}
        if self.default_headers:
            headers.update(self.default_headers)
        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers[
            'X-IBMCloud-SDK-Analytics'] = 'service_name=watson_vision_combined;service_version=V4;operation_id=create_collection'

        params = {'version': self.version}

        form_data = {
            'collection_ids': collection_ids,
            'features': 'objects'
        }
        
        # if images_file:
        #     if not images_filename and hasattr(images_file, 'name'):
        #         images_filename = basename(images_file.name)
        #     form_data['images_file'] = (images_filename, images_file,
        #                                 images_file_content_type or
        #                                 'application/octet-stream')
        # if image_url:
        #     form_data['image_url'] = (None, image_url, 'text/plain')
        if threshold:
            form_data['threshold'] = threshold

        # TODO Use self.request() but we need to sort out how to
        # specify files for FPs
        response = requests.post("{}/v4/analyze".format(self.url),
                         auth=HTTPBasicAuth('apikey', self.iam_apikey),
                         headers=headers,
                         params=params,
                         data=form_data,
                         files={'images_file': image_fp
                               }
                        )
        return wrap_response(response, method='POST')

    
    #########################
    # Collections
    #########################
    
    # https://cloud.ibm.com/apidocs/visual-recognition-v4#create-a-collection
    def create_collection(
            self,
            name=None,
            description="",
            training_status=None,
            **kwargs):
        """Create a collection that can be used to store images.
        
        :param string name: The name of the collection. The name can contain
               alphanumeric, underscore, hyphen, and dot characters. It
               cannot begin with the reserved prefix sys-.
               Constraints: length <= 64, Value must match regular
               expression ^(?!sys-)[\pL\pN_\-.]*$
        """
        headers = {}
        if self.default_headers:
            headers.update(self.default_headers)
        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers[
            'X-IBMCloud-SDK-Analytics'] = 'service_name=watson_vision_combined;service_version=V4;operation_id=create_collection'

        params = {'version': self.version}

        form_data = {}
        if name:
            form_data['name'] = name
            form_data['description'] = description
        if training_status:
            form_data['training_status'] = training_status
        
        response = requests.post("{}/v4/collections".format(self.url),
                         auth=HTTPBasicAuth('apikey', self.iam_apikey),
                         headers=headers,
                         params=params,
                         data=form_data
                        )
        return wrap_response(response, method='POST')

    
    #########################
    # Images
    #########################

    # https://cloud.ibm.com/apidocs/visual-recognition-v4#add-images
    # annotations = [ { 'object': <label>, 'location': {'left':.. , 'height': <bbox height> }}, {}, .. ]
    def add_image(self,
                  collection_id,
                  image_fp,
                  annotations,
                  **kwargs):
        headers = {}
        if self.default_headers:
            headers.update(self.default_headers)
        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers[
            'X-IBMCloud-SDK-Analytics'] = 'service_name=watson_vision_combined;service_version=V4;operation_id=add_image'

        params = {'version': self.version}

        form_data={'training_data': json.dumps({
            'objects': annotations
        }) }
        
        response = requests.post("{}/v4/collections/{}/images".format(self.url, collection_id),
                                 auth=HTTPBasicAuth('apikey', self.iam_apikey),
                                 headers=headers,
                                 params=params,
                                 data=form_data,
                                 files={'images_file': image_fp})
        return wrap_response(response, method='POST')


    def add_image_ltwh(self, collection_id, image_fp, label, ltwh):
        return self.add_image(collection_id, image_fp, [
            create_annotation(label, *ltwh)
        ])
    
    
    # https://cloud.ibm.com/apidocs/visual-recognition-v4#list-images
    def list_images(self,
                    collection_id,
                    **kwargs):
        headers = {}
        if self.default_headers:
            headers.update(self.default_headers)
        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers[
            'X-IBMCloud-SDK-Analytics'] = 'service_name=watson_vision_combined;service_version=V4;operation_id=list_images'

        params = {'version': self.version}
        
        response = requests.get("{}/v4/collections/{}/images".format(self.url, collection_id),
                                auth=HTTPBasicAuth('apikey', self.iam_apikey),
                                headers=headers,
                                params=params)
        return wrap_response(response, method='GET')


    #########################
    # Training
    #########################

    # https://cloud.ibm.com/apidocs/visual-recognition-v4#train-a-collection
    def train(self,
              collection_id,
              **kwargs):
        headers = {}
        if self.default_headers:
            headers.update(self.default_headers)
        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers[
            'X-IBMCloud-SDK-Analytics'] = 'service_name=watson_vision_combined;service_version=V4;operation_id=add_image'

        params = {'version': self.version}

        form_data={'training_data': json.dumps({
            'objects': annotations
        }) }
        
        response = requests.post("{}/v4/collections/{}/train".format(self.url, collection_id),
                                 auth=HTTPBasicAuth('apikey', self.iam_apikey),
                                 headers=headers,
                                 params=params)
        return wrap_response(response, method='POST')
    
    
    # https://console.bluemix.net/apidocs/visual-recognition-v4#get-collection-details
    def get_collection_details(self,
                               collection_id,
                               **kwargs):
        headers = {}
        if self.default_headers:
            headers.update(self.default_headers)
        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['X-IBMCloud-SDK-Analytics'] = 'service_name=watson_vision_combined;service_version=V4;operation_id=get_collection_details'

        params = {'version': self.version}
        
        response = requests.get("{}/v4/collections/{}".format(self.url, collection_id),
                                auth=HTTPBasicAuth('apikey', self.iam_apikey),
                                headers=headers,
                                params=params)
        return wrap_response(response, method='GET')
    
    
    # https://cloud.ibm.com/apidocs/visual-recognition-v4#get-image-details
    def get_image_details(self,
                          collection_id,
                          image_id,
                          **kwargs):
        headers = {}
        if self.default_headers:
            headers.update(self.default_headers)
        if 'headers' in kwargs:
            headers.update(kwargs.get('headers'))
        headers['X-IBMCloud-SDK-Analytics'] = 'service_name=watson_vision_combined;service_version=V4;operation_id=get_image_details'

        params = {'version': self.version}
        
        response = requests.get("{}/v4/collections/{}/images/{}".format(self.url, collection_id, image_id),
                                auth=HTTPBasicAuth('apikey', self.iam_apikey),
                                headers=headers,
                                params=params)
        return wrap_response(response, method='GET')
