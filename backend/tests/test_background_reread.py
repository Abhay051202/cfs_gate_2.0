from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from app.database.models import GateEvent, SystemLog
from app.events.passages import text_can_open_visit
from app.events.cloud_dispatcher import CloudOutboxDispatcher


def seed(repository):
    ts = datetime(2026, 10, 28, 8, 33, 35)
    with repository.db.sessions.begin() as session:
        session.add(GateEvent(id='original', gate_id='gate-in', lane_id='lane-in',
            timestamp=ts, container_number='TGBU3796155', event_type='ENTRY',
            status='CONFIRMED', capture_policy='FINAL_CAPTURE'))
    return SimpleNamespace(id='later', visit_id='later-visit', gate_id='gate-in', lane_id='lane-in',
        direction='ENTRY', started=ts + timedelta(seconds=316),
        body_observed=False, chassis_observed=False)


def test_recovers_actual_316_second_background_reread(repository):
    passage = seed(repository)
    with repository.db.sessions() as session:
        assert repository._recent_text_reread(session, passage, ['TGBU3796155']).id == 'original'


def test_final_capture_save_reuses_original_without_replacing_proof(repository, snapshots, make_job):
    from app.events.passages import Passage, Region
    from app.domain import OCRRead
    from app.ocr.validator import ContainerValidator
    source = seed(repository)
    job = make_job(timestamp=source.started)
    value = ContainerValidator().validate(OCRRead('TGBU3796155', .98))
    region = Region(id='later-text', bbox=(.5, .4, .6, .5))
    region.votes[value.normalized_text] = 1
    region.reads[value.normalized_text] = (value, job)
    region.source_proofs[value.normalized_text] = {job.frame.camera_id: (value, job)}
    passage = Passage(id=source.id, camera_id=job.frame.camera_id, source_id='later-source',
        gate_id=source.gate_id, lane_id=source.lane_id, direction=source.direction,
        started=source.started, last_seen=source.started, track_id=99, bbox=(10, 10, 30, 20),
        regions=[region], final_evidence=True, body_observed=False, chassis_observed=False,
        visit_id=source.visit_id)
    assert repository.save_passage(passage, snapshots, 1) == 'original'
    assert repository.events()['total'] == 1
    assert list(snapshots.root.rglob('*.jpg')) == []


@pytest.mark.parametrize('field,value', [
    ('gate_id', 'gate-out'), ('lane_id', 'lane-2'), ('direction', 'EXIT'),
    ('body_observed', True), ('chassis_observed', True),
    ('started', datetime(2026, 10, 28, 9, 0)),
])
def test_separate_passages_are_not_time_merged(repository, field, value):
    passage = seed(repository)
    setattr(passage, field, value)
    with repository.db.sessions() as session:
        assert repository._recent_text_reread(session, passage, ['TGBU3796155']) is None


@pytest.mark.parametrize('numbers', [['TGBU3796156'], ['TGBU3796155', 'CSQU3054383']])
def test_identity_and_twin_sets_must_match_exactly(repository, numbers):
    passage = seed(repository)
    with repository.db.sessions() as session:
        assert repository._recent_text_reread(session, passage, numbers) is None


def test_background_number_size_cannot_open_new_visit():
    assert not text_can_open_visit((671.6, 299, 725.1, 310.3), (720, 1280, 3), .025)
    assert text_can_open_visit((511, 279, 608, 298), (720, 1280, 3), .025)


def test_duplicate_hidden_and_excluded_without_claiming_cloud_upload(repository, settings):
    passage = seed(repository)
    with repository.db.sessions.begin() as session:
        session.add(GateEvent(id='duplicate', gate_id=passage.gate_id, lane_id=passage.lane_id,
            timestamp=passage.started, container_number='TGBU3796155', event_type='ENTRY',
            status='CONFIRMED', capture_policy='FINAL_CAPTURE'))
    assert repository.mark_duplicate_event('duplicate', 'original')
    repository.strict_ocr_acceptance = True
    repository.enforce_ocr_acceptance('duplicate')
    event = repository.event('duplicate')
    assert event['status'] == 'DUPLICATE'
    assert event['cloud_synced'] is False and event['cloud_synced_at'] is None
    assert repository.events()['total'] == 1
    assert [e['id'] for e in repository.get_unsynced_confirmed_events(include_unreadable=True)] == ['original']
    assert repository.count_unsynced_confirmed_events() == 1
    dispatcher = CloudOutboxDispatcher(settings)
    assert not dispatcher.is_dispatchable_event(event, allow_synced=True)
