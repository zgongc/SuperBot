"""Data Download API endpoints"""
import threading
from flask import request
from ..helpers.response_helper import success_response, error_response
from ..helpers.validation import validate_required_fields


def get_download_service():
    """Get download service from app context"""
    from flask import current_app
    return current_app.download_service


def register_routes(bp):
    """Register data download routes"""

    @bp.route('/data/download/start', methods=['POST'])
    def start_download():
        """POST /api/data/download/start - Start a new download job"""
        try:
            data = request.get_json()

            # Validate required fields
            is_valid, error_msg = validate_required_fields(
                data, ['symbols', 'timeframes', 'start_date']
            )
            if not is_valid:
                return error_response(error_msg, 400)

            symbols = data.get('symbols', [])
            timeframes = data.get('timeframes', [])
            start_date = data.get('start_date')
            end_date = data.get('end_date')  # Optional

            if not symbols:
                return error_response('No symbols provided', 400)

            if not timeframes:
                return error_response('No timeframes provided', 400)

            # Create and start job
            service = get_download_service()
            job = service.create_job(
                symbols=symbols,
                timeframes=timeframes,
                start_date=start_date,
                end_date=end_date
            )

            # Start job in background thread (non-blocking)
            thread = threading.Thread(
                target=service.start_job_sync,
                args=(job.id,),
                daemon=True
            )
            thread.start()

            return success_response(job.to_dict(), message='Download job started')

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/data/download/status/<job_id>', methods=['GET'])
    def get_download_status(job_id):
        """GET /api/data/download/status/<job_id> - Get job status"""
        try:
            service = get_download_service()
            status = service.get_job_status(job_id)

            if status:
                return success_response(status)
            else:
                return error_response('Job not found', 404)

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/data/download/cancel/<job_id>', methods=['POST'])
    def cancel_download(job_id):
        """POST /api/data/download/cancel/<job_id> - Cancel a download job"""
        try:
            service = get_download_service()
            success = service.cancel_job(job_id)

            if success:
                return success_response(message='Job cancelled')
            else:
                return error_response('Job not found', 404)

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/data/download/jobs', methods=['GET'])
    def list_jobs():
        """GET /api/data/download/jobs - List all download jobs"""
        try:
            service = get_download_service()
            jobs = [job.to_dict() for job in service._jobs.values()]

            return success_response({
                'jobs': jobs,
                'total': len(jobs)
            })

        except Exception as e:
            return error_response(str(e), 500)

    @bp.route('/data/download/timeframes', methods=['GET'])
    def get_available_timeframes():
        """GET /api/data/download/timeframes - Get available timeframes"""
        try:
            timeframes = [
                {'value': '1m', 'label': '1 Minute', 'group': 'Minutes'},
                {'value': '3m', 'label': '3 Minutes', 'group': 'Minutes'},
                {'value': '5m', 'label': '5 Minutes', 'group': 'Minutes'},
                {'value': '15m', 'label': '15 Minutes', 'group': 'Minutes'},
                {'value': '30m', 'label': '30 Minutes', 'group': 'Minutes'},
                {'value': '1h', 'label': '1 Hour', 'group': 'Hours'},
                {'value': '2h', 'label': '2 Hours', 'group': 'Hours'},
                {'value': '4h', 'label': '4 Hours', 'group': 'Hours'},
                {'value': '6h', 'label': '6 Hours', 'group': 'Hours'},
                {'value': '8h', 'label': '8 Hours', 'group': 'Hours'},
                {'value': '12h', 'label': '12 Hours', 'group': 'Hours'},
                {'value': '1d', 'label': '1 Day', 'group': 'Days'},
                {'value': '3d', 'label': '3 Days', 'group': 'Days'},
                {'value': '1w', 'label': '1 Week', 'group': 'Weeks'},
                {'value': '1M', 'label': '1 Month', 'group': 'Months'},
            ]

            return success_response({'timeframes': timeframes})

        except Exception as e:
            return error_response(str(e), 500)
