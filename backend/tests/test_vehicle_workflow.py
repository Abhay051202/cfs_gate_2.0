from dataclasses import replace
from datetime import timedelta
from collections import Counter
from unittest.mock import Mock
import numpy as np
from app.domain import Detection, Validation
from app.runtime import GateRuntime
from app.events.passages import PassageCollector, Passage, Region, final_captures
from app.events.cloud_dispatcher import CloudOutboxDispatcher


def setup(settings,repository,snapshots,camera,make_job):
    settings.vehicle_visit_tracking=True
    settings.reversal_grace_seconds=0
    settings.ocr_confirmations=2
    runtime=GateRuntime(settings,[camera],repository,snapshots)
    runtime.detector=Mock()
    runtime.detector.detect_vehicles.return_value=[]
    runtime.model_status['detector']='READY'
    runtime.dispatcher=Mock()
    base=replace(make_job().frame,image=np.random.default_rng(3).integers(0,255,(480,640,3),dtype=np.uint8))
    return runtime,base


def measure(runtime,camera,base,t,boxes):
    f=replace(base,sequence=t+1,timestamp=base.timestamp+timedelta(seconds=t))
    runtime.detector.detect.return_value=[Detection(camera.id,f.timestamp,cls,.95,box) for cls,box in boxes]
    runtime.process_frame(camera,f)


def drain(runtime,size=None):
    while not runtime.queue.empty():
        job=runtime.queue.get_nowait()
        key=runtime.ocr_store.save(job,job.frame.image[20:80,20:100])
        if not job.passage_metadata.get('context_only'):
            n='CSQU3054383' if job.detection.bbox[0]<200 else 'MSCU6639870'
            runtime.ocr_store.save_validation(key,Validation(n,n,.95,True,True,'VALID',size))
        runtime._process_saved_ocr(key)
        runtime.queue.task_done()


def finish(runtime,camera,base):
    for t in range(3,12): measure(runtime,camera,base,t,[])
    runtime._finalize_passages()
    drain(runtime)
    runtime._finalize_passages()


def test_unreadable_departure_has_proof_and_dispatchable_payload(settings,repository,snapshots,camera,make_job):
    r,f=setup(settings,repository,snapshots,camera,make_job)
    measure(r,camera,f,0,[('truck',(5,5,635,479))])
    finish(r,camera,f)
    items=repository.events()['items']
    assert len(items)==1 and items[0]['status']=='UNREADABLE'
    payload=CloudOutboxDispatcher(settings,repository=repository,snapshots=snapshots).build_event_payload(items[0]['id'])
    c=payload['Containers'][0]
    assert c['ContainerNumber'] is None and c['ValidationStatus']=='UNREADABLE'
    assert c['ContainerNumberConfidence']==0 and len(c['Images'])==1
    assert repository.count_unsynced_confirmed_events()==1
    r.dispatcher.enqueue.assert_called_once()


def test_uncertain_two_numbers_split_without_evidence_mixing(settings,repository,snapshots,camera,make_job):
    r,f=setup(settings,repository,snapshots,camera,make_job)
    boxes=[('truck',(5,5,635,479)),('container_number',(30,80,100,120)),('container_number',(430,80,500,120))]
    for t in (0,1): measure(r,camera,f,t,boxes);drain(r)
    finish(r,camera,f)
    items=repository.events()['items']
    assert len(items)==2 and all(not e['is_twin'] for e in items)
    for e in items:
        record=repository.event(e['id'])
        assert {x['normalized_text'] for x in record['ocr_results']}=={e['container_number']}
        evidence=r.ocr_store.event_proof_evidence(e['id'])
        assert {x['normalized_text'] for x in evidence['ocr_results']}=={e['container_number']}
        assert repository.canonical_event_id(e['id'])==e['id']
    assert r.dispatcher.enqueue.call_count==2


def test_supported_twenty_foot_pair_is_one_twin(settings,repository,snapshots,camera,make_job):
    r,f=setup(settings,repository,snapshots,camera,make_job)
    boxes=[('truck',(5,5,635,479)),('container_number',(30,80,100,120)),('container_number',(430,80,500,120))]
    for t in (0,1): measure(r,camera,f,t,boxes);drain(r,'22G1')
    finish(r,camera,f)
    items=repository.events()['items']
    assert len(items)==1 and items[0]['is_twin']
    payload=CloudOutboxDispatcher(settings,repository=repository,snapshots=snapshots).build_event_payload(items[0]['id'])
    assert {c['ContainerNumber'] for c in payload['Containers']}=={'CSQU3054383','MSCU6639870'}
    assert all(len(c['Images'])==1 for c in payload['Containers'])


def test_reversing_reentry_reopens_same_visit(settings,camera,make_job):
    settings.vehicle_visit_tracking=True
    settings.reversal_grace_seconds=30
    c=PassageCollector(settings)
    f=replace(make_job().frame,image=np.full((100,150,3),(20,30,180),dtype=np.uint8))
    body=Detection(camera.id,f.timestamp,'truck',.95,(10,10,140,99))
    c.update(camera,f,[body]);original=next(iter(c.passages.values()))
    for t in range(1,10): c.update(camera,replace(f,timestamp=f.timestamp+timedelta(seconds=t)),[])
    assert original.closed
    returning=replace(f,timestamp=f.timestamp+timedelta(seconds=12),sequence=12)
    c.update(camera,returning,[replace(body,frame_timestamp=returning.timestamp)])
    assert len(c.passages)==1 and not original.closed


def test_next_truck_same_position_different_appearance_is_separate(settings,camera,make_job):
    settings.vehicle_visit_tracking=True
    c=PassageCollector(settings)
    f=replace(make_job().frame,image=np.full((100,150,3),(20,30,180),dtype=np.uint8))
    body=Detection(camera.id,f.timestamp,'truck',.95,(10,10,140,99))
    c.update(camera,f,[body])
    next_frame=replace(f,timestamp=f.timestamp+timedelta(seconds=1),sequence=2,image=np.full((100,150,3),(180,30,20),dtype=np.uint8))
    c.update(camera,next_frame,[replace(body,frame_timestamp=next_frame.timestamp)])
    assert len(c.passages)==2 and len({p.visit_id for p in c.passages.values()})==2


def test_all_matching_camera_proofs_retained(settings,repository,snapshots,make_job,camera):
    ts=make_job().frame.timestamp; group=[]
    cams=[camera.model_copy(update={'id':cam}) for cam in ('camera-1','camera-2','camera-3')]
    repository.sync_cameras(cams)
    for cam in cams:
        j=make_job(camera_id=cam.id,origin='passage:'+cam.id+':number')
        v=Validation('CSQU3054383','CSQU3054383',.95,True,True,'VALID')
        region=Region(votes=Counter({'CSQU3054383':2}),reads={'CSQU3054383':(v,j)},source_proofs={'CSQU3054383':{cam.id:(v,j)}},best_any=(v,j))
        group.append(Passage(cam.id,cam.id,'s','lane-1','lane-1','ENTRY',ts,ts,1,(0,0,150,100),regions=[region],visit_id='visit',closed=True))
    plan=final_captures(group,2)[0];repository.save_passage(plan,snapshots,2)
    payload=CloudOutboxDispatcher(settings,repository=repository,snapshots=snapshots).build_event_payload(plan.id)
    assert {i['CameraId'] for i in payload['Containers'][0]['Images']}=={'camera-1','camera-2','camera-3'}


def test_retry_same_ocr_frame_is_not_an_independent_vote(settings,make_job):
    c=PassageCollector(settings);j=make_job(origin='passage:p:r')
    p=Passage('p','camera-1','s','lane-1','lane-1','ENTRY',j.started_at,j.started_at,1,(0,0,150,100),regions=[Region(id='r')])
    c.passages['p']=p
    v=Validation('CSQU3054383','CSQU3054383',.95,True,True,'VALID')
    c.result('p','r',v,j);c.result('p','r',v,j)
    assert p.regions[0].votes['CSQU3054383']==1


def test_old_camera_activity_cannot_lend_visit_to_new_arrival(settings,camera,make_job):
    settings.vehicle_visit_tracking=True
    c=PassageCollector(settings)
    f=replace(make_job().frame,image=np.full((100,150,3),(20,30,180),dtype=np.uint8))
    body=Detection(camera.id,f.timestamp,'truck',.95,(10,10,140,99))
    c.update(camera,f,[body]);first=next(iter(c.passages.values()))
    other=camera.model_copy(update={'id':'camera-2'})
    later=replace(f,camera_id=other.id,timestamp=f.timestamp+timedelta(minutes=14))
    first.last_seen=later.timestamp
    c.update(other,later,[replace(body,camera_id=other.id,frame_timestamp=later.timestamp)])
    assert len({p.visit_id for p in c.passages.values()})==2


def test_finalization_ids_survive_restart_metadata(settings,repository,snapshots,camera,make_job):
    r,f=setup(settings,repository,snapshots,camera,make_job)
    boxes=[('truck',(5,5,635,479)),('container_number',(30,80,100,120))]
    for t in (0,1):measure(r,camera,f,t,boxes);drain(r)
    before=final_captures(list(r.passages.passages.values()),2)[0].id
    restarted=GateRuntime(settings,[camera],repository,snapshots)
    restarted._recover_passages()
    recovered=final_captures(list(restarted.passages.passages.values()),2)[0].id
    assert recovered==before
    restarted._finalize_passages()
    assert repository.events()['total']==1
    again=GateRuntime(settings,[camera],repository,snapshots)
    again._recover_passages();again._finalize_passages()
    assert repository.events()['total']==1


def test_complete_camera_evidence_publishes_while_truck_waits(settings,repository,snapshots,camera,make_job):
    r,f=setup(settings,repository,snapshots,camera,make_job)
    boxes=[('truck',(5,5,635,479)),('container_number',(30,80,100,120))]
    for t in (0,1): measure(r,camera,f,t,boxes);drain(r)
    r._finalize_passages()
    assert repository.events()['total']==1
    assert len(r.passages.active)==1 and not r.passages.passages
    for t in (2,3,4):measure(r,camera,f,t,boxes)
    assert r.queue.empty()
    r._finalize_passages()
    assert repository.events()['total']==1


def test_published_reversing_truck_returns_without_duplicate(settings,repository,snapshots,camera,make_job):
    r,f=setup(settings,repository,snapshots,camera,make_job)
    settings.reversal_grace_seconds=30
    camera.gate_exit_axis='x';camera.gate_exit_positive=True
    boxes=[('truck',(150,5,635,479)),('container_number',(180,80,240,120))]
    for t in (0,1):measure(r,camera,f,t,boxes);drain(r)
    r._finalize_passages()
    assert repository.events()['total']==1
    moved=[('truck',(10,5,495,479)),('container_number',(30,80,100,120))]
    # Preserve the body's appearance while the measured center reverses.
    constant=replace(f,image=np.full_like(f.image,(20,30,180)))
    for p in r.passages.active.values():
        from app.detection.tracker import appearance
        p.appearance=appearance(constant.image,p.bbox)
    for tracker in r.passages.trackers.values():
        for track in tracker.tracks.values():track.appearance=appearance(constant.image,track.detection.bbox)
    measure(r,camera,constant,2,moved)
    for t in range(3,12):measure(r,camera,constant,t,[])
    assert r.passages.recent
    measure(r,camera,constant,12,moved)
    r._finalize_passages()
    assert repository.events()['total']==1 and r.queue.empty()


def test_number_is_captured_even_when_vehicle_detector_misses(settings,repository,snapshots,camera,make_job):
    r,f=setup(settings,repository,snapshots,camera,make_job)
    boxes=[('container_number',(30,80,100,120))]
    for t in (0,1):measure(r,camera,f,t,boxes);drain(r)
    r._finalize_passages()
    items=repository.events()['items']
    assert len(items)==1 and items[0]['container_number']=='CSQU3054383' and not items[0]['is_twin']


def test_late_vehicle_detection_preserves_number_capture_visit(settings,repository,snapshots,camera,make_job):
    r,f=setup(settings,repository,snapshots,camera,make_job)
    measure(r,camera,f,0,[('container_number',(30,80,100,120))]);drain(r)
    first=next(iter(r.passages.passages))
    measure(r,camera,f,1,[('truck',(5,5,635,479)),('container_number',(30,80,100,120))]);drain(r)
    assert list(r.passages.passages)==[first]
    assert next(iter(r.passages.passages.values())).chassis_observed


def test_unreadable_companion_view_does_not_invent_second_container(settings,make_job):
    j=make_job(origin='passage:p:number')
    v=Validation('CSQU3054383','CSQU3054383',.95,True,True,'VALID')
    known=Region(votes=Counter({'CSQU3054383':2}),reads={'CSQU3054383':(v,j)},best_any=(v,j))
    bad=Validation('partial','',.1,False,False,'INVALID_FORMAT')
    a=Passage('a','camera-1','s','gate','gate','ENTRY',j.started_at,j.started_at,1,(0,0,150,100),regions=[known],visit_id='shared')
    b=replace(a,id='b',camera_id='camera-2',regions=[Region(best_any=(bad,replace(j,origin_key='passage:b:number')))])
    plans=final_captures([a,b],2)
    assert len(plans)==1 and plans[0].regions==[known]


def test_discarded_companion_does_not_resurrect_after_restart(settings,repository,snapshots,camera,make_job):
    r,f=setup(settings,repository,snapshots,camera,make_job)
    other=camera.model_copy(update={'id':'camera-2'})
    repository.sync_cameras([camera,other]);r.cameras.append(other)
    boxes=[('truck',(5,5,635,479)),('container_number',(30,80,100,120))]
    for t in (0,1):
        for cam in (camera,other):measure(r,cam,f,t,boxes)
        while not r.queue.empty():
            job=r.queue.get_nowait();key=r.ocr_store.save(job,job.frame.image[20:80,20:100])
            v=Validation('CSQU3054383','CSQU3054383',.95,True,True,'VALID') if job.detection.camera_id==camera.id else Validation('partial','',.1,False,False,'INVALID_FORMAT')
            r.ocr_store.save_validation(key,v);r._process_saved_ocr(key);r.queue.task_done()
    for t in range(3,12):
        for cam in (camera,other):measure(r,cam,f,t,[])
    r._finalize_passages()
    # Drain remaining independent partial samples retained while OCR ran.
    while not r.queue.empty():
        job=r.queue.get_nowait();key=r.ocr_store.save(job,job.frame.image[20:80,20:100])
        r.ocr_store.save_validation(key,Validation('partial','',.1,False,False,'INVALID_FORMAT'))
        r._process_saved_ocr(key);r.queue.task_done()
    r._finalize_passages()
    assert repository.events()['total']==1
    again=GateRuntime(settings,[camera,other],repository,snapshots)
    again._recover_passages();again._finalize_passages()
    assert repository.events()['total']==1 and not again.passages.passages


def test_vehicle_outside_roi_does_not_keep_visit_alive(settings,camera,make_job):
    settings.vehicle_visit_tracking=True
    camera.occupancy_roi=(0,0,.5,1)
    c=PassageCollector(settings)
    f=replace(make_job().frame,image=np.full((100,150,3),(20,30,180),dtype=np.uint8))
    body=Detection(camera.id,f.timestamp,'truck',.95,(5,5,65,99))
    c.update(camera,f,[body]);p=next(iter(c.passages.values()))
    for t in range(1,10):
        frame=replace(f,timestamp=f.timestamp+timedelta(seconds=t),sequence=t+1)
        c.update(camera,frame,[replace(body,frame_timestamp=frame.timestamp,bbox=(90,5,145,99))])
    assert p.closed


def test_continuous_vehicle_class_change_keeps_one_visit(settings,camera,make_job):
    settings.vehicle_visit_tracking=True
    c=PassageCollector(settings)
    f=replace(make_job().frame,image=np.full((480,640,3),80,dtype=np.uint8))
    for index,cls in enumerate(('truck','vehicle','trailer','truck')):
        frame=replace(f,sequence=index,timestamp=f.timestamp+timedelta(seconds=index*.2))
        c.update(camera,frame,[Detection(camera.id,frame.timestamp,cls,.95,(5,5,635,479))])
    assert len(c.passages)==1
    assert len(c.active)==1

def test_continuous_box_lighting_change_keeps_track():
    from app.detection.tracker import TemporalTracker
    from app.domain import utcnow
    t=TemporalTracker();now=utcnow()
    d=Detection('cam',now,'truck',.95,(5,5,95,95))
    first=t.update([d],now,np.full((100,100,3),80,dtype=np.uint8))[0].id
    second=t.update([d],now+timedelta(seconds=.2),np.full((100,100,3),140,dtype=np.uint8))[0].id
    assert first==second


def test_recent_camera_joins_visit_despite_stale_track_alias(settings,camera,make_job):
    settings.vehicle_visit_tracking=True
    c=PassageCollector(settings)
    f=replace(make_job().frame,image=np.full((480,640,3),80,dtype=np.uint8))
    body=Detection(camera.id,f.timestamp,'truck',.95,(5,5,635,479))
    c.update(camera,f,[body])
    first=next(iter(c.passages.values()))
    stale=replace(first,id='stale',visit_id='old',track_id=99,
                  started=f.timestamp-timedelta(seconds=100),last_seen=f.timestamp-timedelta(seconds=20))
    c.active[(camera.id,f.source_id,99)]=stale
    other=camera.model_copy(update={'id':'camera-2'})
    frame=replace(f,camera_id=other.id,timestamp=f.timestamp+timedelta(seconds=1))
    c.update(other,frame,[replace(body,camera_id=other.id,frame_timestamp=frame.timestamp)])
    assert {p.visit_id for p in c.passages.values()}=={first.visit_id}
    assert len(c.groups())==1


def test_following_truck_cannot_borrow_other_camera_visit(settings,camera,make_job):
    settings.vehicle_visit_tracking=True
    c=PassageCollector(settings)
    f=replace(make_job().frame,image=np.full((480,640,3),(20,30,180),dtype=np.uint8))
    body=Detection(camera.id,f.timestamp,'truck',.95,(5,5,635,479))
    c.update(camera,f,[body])
    first=next(iter(c.passages.values()))
    other=camera.model_copy(update={'id':'camera-2'})
    second_view=replace(f,camera_id=other.id,timestamp=f.timestamp+timedelta(seconds=.2))
    c.update(other,second_view,[replace(body,camera_id=other.id,frame_timestamp=second_view.timestamp)])
    incoming=replace(f,timestamp=f.timestamp+timedelta(seconds=1),image=np.full((480,640,3),(180,30,20),dtype=np.uint8))
    c.update(camera,incoming,[replace(body,frame_timestamp=incoming.timestamp)])
    assert len({p.visit_id for p in c.passages.values()})==2
    assert len(c.groups())==2


def test_shared_visit_group_not_split_by_late_view_start(settings,make_job):
    c=PassageCollector(settings);ts=make_job().frame.timestamp
    for index in range(2):
        p=Passage(str(index),'camera-'+str(index),'s','gate','lane','ENTRY',
                  ts+timedelta(seconds=60*index),ts,1,(0,0,150,100),visit_id='shared')
        c.passages[p.id]=p
    assert len(c.groups())==1


def test_matching_number_cannot_bridge_two_same_camera_visits(settings,make_job):
    c=PassageCollector(settings);ts=make_job().frame.timestamp
    for identity,cam,visit in [('a','camera-1','first'),('b','camera-2','first'),('c','camera-1','next')]:
        region=Region(votes=Counter({'CSQU3054383':2}))
        c.passages[identity]=Passage(identity,cam,'s','gate','lane','ENTRY',ts,ts,1,
                                   (0,0,150,100),regions=[region],visit_id=visit)
    assert len(c.groups())==2


def test_low_confidence_railing_prediction_cannot_create_visit(settings,camera,make_job):
    settings.vehicle_visit_tracking=True
    c=PassageCollector(settings)
    f=replace(make_job().frame,image=np.full((1440,2560,3),80,dtype=np.uint8))
    for index in range(15):
        frame=replace(f,sequence=index,timestamp=f.timestamp+timedelta(seconds=index))
        false_truck=Detection(camera.id,frame.timestamp,'truck',.2719978392,
                              (875.2914,807.2139,1896.6328,1096.729))
        assert c.update(camera,frame,[false_truck])==[]
    assert not c.passages and not c.active


def test_independent_matching_single_reads_join_camera_results(settings,repository,snapshots,make_job,camera):
    settings.vehicle_visit_tracking=True
    c=PassageCollector(settings);ts=make_job().frame.timestamp
    cams=[camera.model_copy(update={'id':name}) for name in ('camera-1','camera-2')]
    repository.sync_cameras(cams)
    for index,cam in enumerate(cams):
        job=make_job(camera_id=cam.id,origin='passage:'+cam.id+':number')
        v=Validation('CSQU3054383','CSQU3054383',.95,True,True,'VALID')
        region=Region(votes=Counter({'CSQU3054383':1}),reads={'CSQU3054383':(v,job)},
                      source_proofs={'CSQU3054383':{cam.id:(v,job)}},best_any=(v,job))
        c.passages[cam.id]=Passage(cam.id,cam.id,'s',camera.gate_id,camera.gate_id,'ENTRY',
                                 ts+timedelta(seconds=index),ts+timedelta(seconds=2),1,
                                 (0,0,150,100),regions=[region],visit_id=cam.id,closed=True)
    groups=c.groups()
    assert len(groups)==1
    plans=final_captures(groups[0],2)
    assert len(plans)==1
    repository.save_passage(plans[0],snapshots,2)
    item=repository.event(plans[0].id)
    assert item['status']=='CONFIRMED'
    assert {proof['camera_id'] for proof in item['snapshots']}=={'camera-1','camera-2'}


def test_matching_reads_without_temporal_overlap_stay_separate(settings,make_job):
    c=PassageCollector(settings);ts=make_job().frame.timestamp
    for index in range(2):
        region=Region(votes=Counter({'CSQU3054383':1}))
        start=ts+timedelta(seconds=index*20)
        c.passages[str(index)]=Passage(str(index),'camera-'+str(index),'s','gate','lane','ENTRY',
                                     start,start,1,(0,0,150,100),regions=[region],visit_id=str(index))
    assert len(c.groups())==2


def test_context_recovery_uses_saved_body_and_persists_text_crop(settings,repository,snapshots,camera,make_job):
    r,f=setup(settings,repository,snapshots,camera,make_job)
    body=Detection(camera.id,f.timestamp,'truck',.9,(100,100,500,400))
    job=replace(make_job(),frame=f,detection=body,event_id='visit',passage_id='p',region_id='r',
                passage_metadata={'context_only':True,'lane_id':camera.gate_id,'bbox':body.bbox})
    text=Detection(camera.id,f.timestamp,'container_number',.9,(30,40,150,80))
    r.detector.detect.return_value=[text]
    recovered,crop=r._recover_context_number(job)
    assert recovered.detection.bbox==(130,140,250,180)
    assert crop.shape[:2]==(40,120)
    assert r.detector.detect.call_args.args[0].image.shape[:2]==(300,400)
    key=r.ocr_store.save(job,f.image[100:400,100:500])
    r.ocr_store.retarget_context(key,recovered,crop)
    loaded,saved,_=r.ocr_store.load(key)
    assert loaded.detection.bbox==recovered.detection.bbox
    assert not loaded.passage_metadata['context_only']
    assert np.array_equal(saved,crop)


def test_context_recovery_rejects_multiple_container_numbers(settings,repository,snapshots,camera,make_job):
    r,f=setup(settings,repository,snapshots,camera,make_job)
    job=replace(make_job(),frame=f,detection=Detection(camera.id,f.timestamp,'truck',.9,(0,0,640,480)))
    r.detector.detect.return_value=[Detection(camera.id,f.timestamp,'container_number',.9,b)
                                    for b in [(30,40,150,80),(300,40,450,80)]]
    assert r._recover_context_number(job) is None


def test_context_recovery_rejects_number_outside_saved_vehicle(settings,repository,snapshots,camera,make_job):
    r,f=setup(settings,repository,snapshots,camera,make_job)
    job=replace(make_job(),frame=f,detection=Detection(camera.id,f.timestamp,'truck',.9,(100,100,400,300)))
    r.detector.detect.return_value=[Detection(camera.id,f.timestamp,'container_number',.9,(280,20,350,50))]
    assert r._recover_context_number(job) is None


def test_tiny_vehicle_prediction_cannot_create_context_event(settings,camera,make_job):
    settings.vehicle_visit_tracking=True
    collector=PassageCollector(settings)
    frame=replace(make_job().frame,image=np.zeros((1440,2560,3),dtype=np.uint8))
    false_body=Detection(camera.id,frame.timestamp,'truck',.9,(100,100,218,122))
    collector.update(camera,frame,[false_body])
    assert not collector.passages


def test_shared_visit_folder_stays_stable_across_cameras_and_finalization(settings,repository,snapshots,camera,make_job):
    settings.shared_visit_folders=True
    settings.container_roi_trigger=True
    r,f=setup(settings,repository,snapshots,camera,make_job)
    other=camera.model_copy(update={'id':'camera-2'})
    r.cameras=[camera,other];repository.sync_cameras(r.cameras)
    boxes=[('truck',(5,5,635,479)),('container_number',(30,80,100,120))]
    paths=[]
    for t in (0,1):
        for cam in r.cameras:
            measure(r,cam,replace(f,camera_id=cam.id),t,boxes);drain(r)
    visits={p.visit_id for p in r.passages.passages.values()}
    assert len(visits)==1
    visit=next(iter(visits))
    with r.ocr_store.connection() as db:
        paths=[row['image'] for row in db.execute('select image from jobs')]
    assert all(path.startswith(visit+'/') and len(path.split('/'))==2 for path in paths)
    r._finalize_passages()
    items=repository.events()['items']
    assert len(items)==1 and items[0]['visit_id']==visit
    record=repository.event(items[0]['id'])
    assert {p['camera_id'] for p in record['snapshots']}=={camera.id,other.id}
    assert all(p['image_path'].startswith(visit+'/') and len(p['image_path'].split('/'))==2 for p in record['snapshots'])
    with r.ocr_store.connection() as db:
        assert paths==[row['image'] for row in db.execute('select image from jobs')]
    assert {p.name for p in snapshots.root.iterdir() if p.is_dir()}=={visit}
    assert (snapshots.root/visit/'visit.json').is_file()
    restarted=GateRuntime(settings,r.cameras,repository,snapshots)
    restarted._recover_passages()
    assert not restarted.passages.passages
    assert {p.name for p in snapshots.root.iterdir() if p.is_dir()}=={visit}


def test_two_container_records_keep_one_visit_folder(settings,repository,snapshots,camera,make_job):
    settings.shared_visit_folders=True;settings.container_roi_trigger=True
    r,f=setup(settings,repository,snapshots,camera,make_job)
    boxes=[('truck',(5,5,635,479)),('container_number',(30,80,100,120)),('container_number',(430,80,500,120))]
    for t in (0,1):measure(r,camera,f,t,boxes);drain(r)
    visit=next(iter(r.passages.passages.values())).visit_id
    r._finalize_passages()
    items=repository.events()['items']
    assert len(items)==2 and {item['visit_id'] for item in items}=={visit}
    assert {p.name for p in snapshots.root.iterdir() if p.is_dir()}=={visit}
    import json
    manifest=json.loads((snapshots.root/visit/'visit.json').read_text())
    assert set(manifest['container_records'])=={item['id'] for item in items}
    for item in items:
        payload=CloudOutboxDispatcher(settings,repository=repository,snapshots=snapshots).build_event_payload(item['id'])
        assert payload['Containers'][0]['ContainerNumber']==item['container_number']
        assert len(payload['Containers'][0]['Images'])==1


def test_container_roi_is_required_to_start_visit(settings,camera,make_job):
    settings.vehicle_visit_tracking=True;settings.container_roi_trigger=True
    collector=PassageCollector(settings)
    f=replace(make_job().frame,image=np.zeros((480,640,3),dtype=np.uint8))
    body=Detection(camera.id,f.timestamp,'truck',.95,(5,5,635,479))
    collector.update(camera,f,[body])
    assert not collector.passages
    container=Detection(camera.id,f.timestamp,'container',.9,(10,20,300,300))
    collector.update(camera,replace(f,sequence=2,timestamp=f.timestamp+timedelta(seconds=1)),[body,container])
    assert len(collector.passages)==1


def test_container_body_owns_its_number_without_truck_model(settings,camera,make_job):
    settings.vehicle_visit_tracking=True;settings.container_roi_trigger=True
    collector=PassageCollector(settings)
    f=replace(make_job().frame,image=np.random.default_rng(1).integers(0,255,(480,640,3),dtype=np.uint8))
    detections=[Detection(camera.id,f.timestamp,cls,.95,box) for cls,box in
                [('container',(10,10,500,450)),('container_number',(50,50,200,100))]]
    collector.update(camera,f,detections)
    assert len(collector.passages)==1
    passage=next(iter(collector.passages.values()))
    assert len(passage.regions)==1 and not passage.chassis_observed


def test_shared_visit_recovery_keeps_camera_association(settings,repository,snapshots,camera,make_job):
    settings.shared_visit_folders=True;settings.container_roi_trigger=True
    r,f=setup(settings,repository,snapshots,camera,make_job)
    other=camera.model_copy(update={'id':'camera-2'})
    r.cameras=[camera,other];repository.sync_cameras(r.cameras)
    boxes=[('truck',(5,5,635,479)),('container_number',(30,80,100,120))]
    for cam in r.cameras:measure(r,cam,replace(f,camera_id=cam.id),0,boxes);drain(r)
    visit=next(iter(r.passages.passages.values())).visit_id
    restarted=GateRuntime(settings,r.cameras,repository,snapshots)
    restarted._recover_passages()
    assert {p.visit_id for p in restarted.passages.passages.values()}=={visit}
    assert len(restarted.passages.groups())==1
    restarted._finalize_passages()
    assert len(repository.events()['items'])==1
    assert {p.name for p in snapshots.root.iterdir() if p.is_dir()}=={visit}


def test_shared_folder_created_at_roi_trigger_before_ocr(settings,repository,snapshots,camera,make_job):
    settings.shared_visit_folders=True;settings.container_roi_trigger=True
    r,f=setup(settings,repository,snapshots,camera,make_job)
    measure(r,camera,f,0,[('truck',(5,5,635,479)),('container_number',(30,80,100,120))])
    visit=next(iter(r.passages.passages.values())).visit_id
    assert (snapshots.root/visit).is_dir()
    assert not r.queue.empty()
    assert not repository.events()['items']


def test_roi_visit_collects_every_fresh_gate_camera_in_flat_folder(settings,repository,snapshots,camera,make_job):
    settings.shared_visit_folders=True;settings.container_roi_trigger=True
    r,f=setup(settings,repository,snapshots,camera,make_job)
    cams=[camera,camera.model_copy(update={'id':'camera-2'}),camera.model_copy(update={'id':'camera-3'})]
    exit_camera=camera.model_copy(update={'id':'exit','gate_id':'other','direction':'EXIT'})
    r.cameras=cams+[exit_camera];repository.sync_cameras(r.cameras)
    for cam in cams[1:]+[exit_camera]:
        r.manager.workers[cam.id]=Mock()
        r.manager.workers[cam.id].latest.return_value=replace(f,camera_id=cam.id)
    boxes=[('truck',(5,5,635,479)),('container_number',(30,80,100,120))]
    measure(r,camera,f,0,boxes)
    visit=next(iter(r.passages.passages.values())).visit_id
    folder=snapshots.root/visit
    assert {p.name for p in folder.glob('*_context_0.jpg')}=={cam.id+'_context_0.jpg' for cam in cams}
    assert not list(folder.iterdir())[0].is_dir()
    assert not list(folder.glob('exit*'))
    for t in range(1,8):
        for cam in cams[1:]:r.manager.workers[cam.id].latest.return_value=replace(f,camera_id=cam.id,timestamp=f.timestamp+timedelta(seconds=t))
        measure(r,camera,f,t,boxes)
    assert len(list(folder.glob('*_context_*.jpg')))==9
    assert not any(p.is_dir() for p in folder.iterdir())


def test_gate_context_rejects_stale_camera_frames(settings,repository,snapshots,camera,make_job):
    settings.shared_visit_folders=True;settings.container_roi_trigger=True
    r,f=setup(settings,repository,snapshots,camera,make_job)
    other=camera.model_copy(update={'id':'camera-2'});r.cameras=[camera,other]
    r.manager.workers[other.id]=Mock()
    r.manager.workers[other.id].latest.return_value=replace(f,camera_id=other.id,timestamp=f.timestamp-timedelta(seconds=60))
    measure(r,camera,f,0,[('truck',(5,5,635,479)),('container_number',(30,80,100,120))])
    visit=next(iter(r.passages.passages.values())).visit_id
    assert not list((snapshots.root/visit).glob('camera-2_context*'))


def test_same_number_independent_frames_across_regions_confirm_one_visit(settings,repository,snapshots,camera,make_job):
    r,f=setup(settings,repository,snapshots,camera,make_job)
    measure(r,camera,f,0,[('truck',(5,5,635,479)),('container_number',(30,80,100,120))]);drain(r)
    measure(r,camera,f,1,[('truck',(5,5,635,479)),('container_number',(130,180,190,220))]);drain(r)
    assert len(next(iter(r.passages.passages.values())).regions)==2
    finish(r,camera,f)
    items=repository.events()['items']
    assert len(items)==1 and items[0]['status']=='CONFIRMED'


def test_one_camera_number_proof_is_previewed_but_upload_is_held(settings,repository,snapshots,camera,make_job):
    r,f=setup(settings,repository,snapshots,camera,make_job)
    settings.minimum_number_proof_cameras=2
    boxes=[('truck',(5,5,635,479)),('container_number',(30,80,100,120))]
    for t in (0,1):measure(r,camera,f,t,boxes);drain(r)
    r._finalize_passages()
    assert not repository.events()['items']
    finish(r,camera,f)
    item=repository.events()['items'][0]
    dispatcher=CloudOutboxDispatcher(settings,repository=repository,snapshots=snapshots)
    preview=dispatcher.build_proof_preview(item['id'])
    assert len(preview['Containers'][0]['Images'])==1
    import pytest,base64
    with pytest.raises(ValueError,match='1 of 2 cameras'):
        dispatcher.build_event_payload(item['id'])
    proof=repository.event(item['id'])['snapshots'][0]
    data=base64.b64decode(preview['Containers'][0]['Images'][0]['Image'].split(',',1)[1])
    assert data==snapshots.path(proof['image_path']).read_bytes()


def test_two_camera_preview_and_upload_images_are_identical(settings,repository,snapshots,make_job,camera):
    settings.minimum_number_proof_cameras=2
    group=[];ts=make_job().frame.timestamp
    cams=[camera,camera.model_copy(update={'id':'camera-2'})];repository.sync_cameras(cams)
    for cam in cams:
        job=make_job(camera_id=cam.id,origin='passage:'+cam.id+':number')
        validation=Validation('CSQU3054383','CSQU3054383',.95,True,True,'VALID')
        region=Region(votes=Counter({'CSQU3054383':2}),reads={'CSQU3054383':(validation,job)},
                      source_proofs={'CSQU3054383':{cam.id:(validation,job)}},best_any=(validation,job))
        group.append(Passage(cam.id,cam.id,'s','lane-1','lane-1','ENTRY',ts,ts,1,(0,0,150,100),
                             regions=[region],visit_id='visit',closed=True))
    plan=final_captures(group,2)[0];repository.save_passage(plan,snapshots,2)
    dispatcher=CloudOutboxDispatcher(settings,repository=repository,snapshots=snapshots)
    assert dispatcher.build_proof_preview(plan.id)==dispatcher.build_event_payload(plan.id)
    assert len(dispatcher.build_event_payload(plan.id)['Containers'][0]['Images'])==2


def test_two_labels_during_brief_body_dropout_keep_one_visit(settings,camera,make_job):
    settings.vehicle_visit_tracking = True
    settings.container_roi_trigger = True
    collector = PassageCollector(settings)
    base = replace(make_job().frame,image=np.full((100,150,3),(180,60,30),dtype=np.uint8))
    body = Detection(camera.id,base.timestamp,'container',.9,(5,5,145,99))
    first = replace(body,class_name='container_number',bbox=(20,20,40,35))
    collector.update(camera,base,[body,first])
    passage = next(iter(collector.passages.values()))
    context = passage.context
    for i, box in enumerate([(20,20,40,35),(100,60,120,80)],1):
        frame = replace(base,sequence=i+1,timestamp=base.timestamp+timedelta(seconds=.2*i))
        collector.update(camera,frame,[replace(first,frame_timestamp=frame.timestamp,bbox=box)])
    assert len(collector.passages) == 1
    assert passage.bbox == body.bbox
    assert passage.body_observed and passage.context is context
    assert len(passage.regions) == 2  # Distinct physical labels keep their geometry.
    assert passage.last_body_seen == base.timestamp  # Text cannot extend the timeout.


def test_body_memory_does_not_absorb_changed_or_late_container(settings,camera,make_job):
    settings.vehicle_visit_tracking = True
    for age, color in [(1,(20,30,180)),(3,(180,60,30))]:
        collector = PassageCollector(settings)
        base = replace(make_job().frame,image=np.full((100,150,3),(180,60,30),dtype=np.uint8))
        body = Detection(camera.id,base.timestamp,'container',.9,(5,5,145,99))
        collector.update(camera,base,[body])
        frame = replace(base,sequence=2,timestamp=base.timestamp+timedelta(seconds=age),
                        image=np.full((100,150,3),color,dtype=np.uint8))
        collector.update(camera,frame,[replace(body,class_name='container_number',bbox=(100,60,120,80),frame_timestamp=frame.timestamp)])
        assert len(collector.passages) == 2


def test_companion_joins_before_detection_and_queues_roi_recovery(settings,repository,snapshots,camera,make_job):
    settings.shared_visit_folders=True;settings.container_roi_trigger=True
    r,f=setup(settings,repository,snapshots,camera,make_job)
    other=camera.model_copy(update={'id':'camera-2'})
    r.cameras=[camera,other];repository.sync_cameras(r.cameras)
    r.manager.workers[other.id]=Mock()
    r.manager.workers[other.id].latest.return_value=replace(f,camera_id=other.id)
    measure(r,camera,f,0,[('truck',(5,5,635,479)),('container_number',(30,80,100,120))])
    members=list(r.passages.passages.values())
    assert len(members)==2 and len({p.visit_id for p in members})==1
    companion=next(p for p in members if p.camera_id==other.id)
    assert companion.capture_only
    queued=list(r.queue.queue)
    recovery=next(j for j in queued if j.detection.camera_id==other.id)
    assert recovery.passage_metadata['context_only']
    assert recovery.event_id==members[0].visit_id
    assert recovery.detection.class_name=='capture_roi'
    before=companion.id
    measure(r,other,replace(f,camera_id=other.id),1,
            [('truck',(5,5,635,479)),('container_number',(430,80,500,120))])
    assert len(r.passages.passages)==2
    assert r.passages.passages[before] is companion and not companion.capture_only
    assert len({p.visit_id for p in r.passages.passages.values()})==1


def test_companion_recovery_is_bounded_and_cannot_hold_visit_open(settings,repository,snapshots,camera,make_job):
    settings.shared_visit_folders=True;settings.container_roi_trigger=True
    r,f=setup(settings,repository,snapshots,camera,make_job)
    other=camera.model_copy(update={'id':'camera-2'})
    r.cameras=[camera,other];repository.sync_cameras(r.cameras)
    r.manager.workers[other.id]=Mock()
    for t in range(6):
        r.manager.workers[other.id].latest.return_value=replace(f,camera_id=other.id,sequence=t+1,timestamp=f.timestamp+timedelta(seconds=t))
        measure(r,camera,f,t,[('truck',(5,5,635,479)),('container_number',(30,80,100,120))])
    companion=next(p for p in r.passages.passages.values() if p.camera_id==other.id)
    region=companion.regions[0]
    assert len(region.capture_sequences)==3
    assert region.pending+len(region.candidates)==3
    for t in range(6,15):
        measure(r,camera,f,t,[])
    assert companion.closed
    assert all(p.closed for p in r.passages.passages.values())


def test_companion_reservation_does_not_adopt_two_simultaneous_targets(settings,camera,make_job):
    settings.vehicle_visit_tracking=True;settings.shared_visit_folders=True
    c=PassageCollector(settings)
    f=replace(make_job().frame,image=np.random.default_rng(9).integers(0,255,(100,150,3),dtype=np.uint8))
    c.update(camera,f,[Detection(camera.id,f.timestamp,'truck',.95,(0,0,149,99))])
    owner=next(iter(c.passages.values()))
    other=camera.model_copy(update={'id':'camera-2'})
    g=replace(f,camera_id=other.id)
    reserved=c.reserve_observer(owner,other,g)
    c.update(other,g,[Detection(other.id,g.timestamp,'truck',.95,b) for b in [(0,0,65,99),(80,0,149,99)]])
    assert reserved.capture_only
    actual=[p for p in c.active.values() if p.camera_id==other.id]
    assert len(actual)==2
    # A reserved observer must not guess which simultaneous truck it belongs to.
    assert all(p.id!=reserved.id and p.visit_id!=owner.visit_id for p in actual)


def test_companion_frame_before_visit_is_not_collected(settings,repository,snapshots,camera,make_job):
    settings.shared_visit_folders=True;settings.container_roi_trigger=True
    r,f=setup(settings,repository,snapshots,camera,make_job)
    other=camera.model_copy(update={'id':'camera-2'});r.cameras=[camera,other]
    r.manager.workers[other.id]=Mock()
    r.manager.workers[other.id].latest.return_value=replace(f,camera_id=other.id,timestamp=f.timestamp-timedelta(seconds=1))
    measure(r,camera,f,0,[('truck',(5,5,635,479)),('container_number',(30,80,100,120))])
    assert len(r.passages.passages)==1
    assert all(j.detection.camera_id==camera.id for j in r.queue.queue)


def test_companion_recovery_reaches_same_event_and_two_camera_proofs(settings,repository,snapshots,camera,make_job):
    from app.domain import OCRRead
    settings.shared_visit_folders=True;settings.container_roi_trigger=True
    settings.minimum_number_proof_cameras=2
    r,f=setup(settings,repository,snapshots,camera,make_job)
    other=camera.model_copy(update={'id':'camera-2'})
    r.cameras=[camera,other];repository.sync_cameras(r.cameras)
    r.manager.workers[other.id]=Mock()
    r._read_ocr=Mock(return_value=OCRRead('CSQU3054383',.95))
    for t in range(2):
        other_frame=replace(f,camera_id=other.id,sequence=100+t,timestamp=f.timestamp+timedelta(seconds=t))
        r.manager.workers[other.id].latest.return_value=other_frame
        measure(r,camera,f,t,[('truck',(5,5,635,479)),('container_number',(30,80,100,120))])
        while not r.queue.empty():
            job=r.queue.get_nowait()
            a,b,c,d=map(int,job.detection.bbox)
            key=r.ocr_store.save(job,job.frame.image[b:d,a:c])
            r.detector.detect.return_value=[Detection(other.id,other_frame.timestamp,'container_number',.95,(430,80,500,120))]
            r._process_saved_ocr(key)
            r.queue.task_done()
    r._finalize_passages()
    items=repository.events()['items']
    assert len(items)==1 and items[0]['container_number']=='CSQU3054383'
    payload=CloudOutboxDispatcher(settings,repository=repository,snapshots=snapshots).build_event_payload(items[0]['id'])
    assert {image['CameraId'] for image in payload['Containers'][0]['Images']}=={camera.id,other.id}
    assert len([p for p in snapshots.root.iterdir() if p.is_dir()])==1


def test_matching_text_fragments_reconcile_but_measured_trucks_do_not(settings,camera,make_job):
    settings.vehicle_visit_tracking=True
    c=PassageCollector(settings);ts=make_job().frame.timestamp
    def fragment(identity,sequence,number='CSQU3054383',body=False,seconds=0):
        job=make_job(sequence=sequence,timestamp=ts+timedelta(seconds=seconds))
        v=Validation(number,number,.95,True,True,'VALID')
        region=Region(votes=Counter({number:1}),reads={number:(v,job)},best_any=(v,job))
        p=Passage(identity,camera.id,job.frame.source_id,'gate','lane','ENTRY',job.frame.timestamp,
                  job.frame.timestamp,sequence,(10,10,100,30),regions=[region],visit_id=identity,
                  body_observed=body,chassis_observed=body)
        return p
    a,b=fragment('first',1),fragment('second',2,seconds=1)
    c.passages={p.id:p for p in (a,b)}
    c.reconcile_text_visits()
    assert a.visit_id==b.visit_id and len(c.groups())==1
    for negative in [fragment('other-number',3,'MSCU6639870'),
                     fragment('actual-truck',3,body=True),
                     fragment('later-visit',3,seconds=20),fragment('same-frame',1)]:
        assert not c._same_text_fragment(a,negative)
    closed=fragment('closed',4)
    closed.closed=True;closed.closed_at=ts+timedelta(seconds=.5)
    assert not c._same_text_fragment(closed,b)
    changed=fragment('changed-source',5);changed.source_id='new-camera-generation'
    assert not c._same_text_fragment(a,changed)


def test_repeated_markings_finish_as_one_container_and_one_folder(settings,repository,snapshots,camera,make_job):
    from app.domain import OCRRead
    settings.shared_visit_folders=True;settings.container_roi_trigger=True
    r,f=setup(settings,repository,snapshots,camera,make_job)
    r._read_ocr=Mock(return_value=OCRRead('CSQU3054383',.95))
    boxes=[('container_number',(30,80,100,120)),('container_number',(430,80,500,120))]
    for t in (0,1):
        measure(r,camera,f,t,boxes)
        while not r.queue.empty():
            job=r.queue.get_nowait()
            a,b,c,d=map(int,job.detection.bbox)
            key=r.ocr_store.save(job,job.frame.image[b:d,a:c])
            r._process_saved_ocr(key);r.queue.task_done()
    assert len({p.visit_id for p in r.passages.passages.values()})==1
    r._finalize_passages()
    events=repository.events()['items']
    assert len(events)==1 and events[0]['container_number']=='CSQU3054383'
    assert len([p for p in snapshots.root.iterdir() if p.is_dir()])==1


def test_text_reconciliation_follows_continuous_chain_not_total_time_window(settings,make_job,camera):
    settings.vehicle_visit_tracking=True
    c=PassageCollector(settings);ts=make_job().frame.timestamp
    for i in range(4):
        job=make_job(sequence=i+1,timestamp=ts+timedelta(seconds=1.5*i))
        v=Validation('CSQU3054383','CSQU3054383',.95,True,True,'VALID')
        region=Region(votes=Counter({v.normalized_text:1}),reads={v.normalized_text:(v,job)})
        p=Passage(str(i),camera.id,job.frame.source_id,'g','l','ENTRY',job.frame.timestamp,
                  job.frame.timestamp,i,(10,10,100,30),regions=[region],visit_id=str(i),
                  body_observed=False,chassis_observed=False)
        c.passages[p.id]=p
    assert len(c.groups())==1


def test_moving_container_text_and_companion_camera_finish_as_one_event(settings,repository,snapshots,camera,make_job):
    settings.shared_visit_folders=True
    settings.container_roi_trigger=True
    settings.ocr_confirmations=1
    settings.minimum_number_proof_cameras=1
    cam2 = camera.model_copy(update={'id': 'gate-in-3', 'role': 'REAR'})
    repository.sync_cameras([camera, cam2])
    r, f = setup(settings, repository, snapshots, camera, make_job)
    r.cameras = [camera, cam2]
    # Frame 0 on cam1: moving box at x=100
    measure(r, camera, f, 0, [('container_number', (100, 80, 150, 200))])
    # Frame 1 on cam1: container moves to x=160
    measure(r, camera, f, 1, [('container_number', (160, 80, 210, 200))])
    # Frame 2 on cam2 (rear): container seen at x=120
    measure(r, cam2, f, 2, [('container_number', (120, 80, 180, 200))])
    drain(r)
    for t in range(4, 15):
        measure(r, camera, f, t, [])
        measure(r, cam2, f, t, [])
    r._finalize_passages()
    drain(r)
    r._finalize_passages()
    events = repository.events()['items']
    assert len(events) == 1
    assert events[0]['container_number'] in ('HASU5106563', 'CSQU3054383')
    assert events[0]['status'] in ('CONFIRMED', 'CAPTURED')
    event_detail = repository.event(events[0]['id'])
    cameras_in_proof = {s['camera_id'] for s in event_detail['snapshots']}
    assert cameras_in_proof == {'camera-1', 'gate-in-3'}


def test_early_finalization_on_gate_stop_and_late_second_camera_override(settings, repository, snapshots, camera, make_job):
    """When a truck stops at a multi-camera gate for security checking, confirm and dispatch immediately
    as soon as 1 camera reads a valid check-digit container number, then override images when Camera 2 reads it."""
    settings.shared_visit_folders = True
    settings.container_roi_trigger = True
    settings.ocr_confirmations = 1
    settings.minimum_number_proof_cameras = 1
    cam1 = camera.model_copy(update={'id': 'gate-out-2', 'role': 'FRONT', 'gate_id': 'lane-out', 'direction_by_heading': 'EXIT'})
    cam2 = camera.model_copy(update={'id': 'gate-out-1', 'role': 'REAR', 'gate_id': 'lane-out', 'direction_by_heading': 'EXIT'})
    repository.sync_cameras([cam1, cam2])
    r, f = setup(settings, repository, snapshots, cam1, make_job)
    f = replace(f, camera_id=cam1.id)
    r.cameras = [cam1, cam2]

    # Step 1: Truck arrives at cam1 and stops for checking (cam2 has NOT seen the number yet, truck has NOT departed)
    measure(r, cam1, f, 0, [('truck', (20, 20, 600, 460)), ('container_number', (100, 80, 180, 140))])
    drain(r)
    r._finalize_passages()

    # Event must be finalized immediately without waiting for cam2 or physical departure
    events = repository.events()['items']
    assert len(events) == 1
    ev_id = events[0]['id']
    assert events[0]['container_number'] == 'CSQU3054383'
    assert events[0]['status'] == 'CONFIRMED'
    assert r.dispatcher.enqueue.call_count == 1

    # Simulate Step 1 cloud sync completing with cloud_event_id
    repository.mark_cloud_synced(ev_id, cloud_event_id='0e7d4fac-5f22-48a4-be37-0bc58a948eff')
    assert repository.event(ev_id)['cloud_synced'] is True

    # Step 2: 15 seconds later, truck moves forward after security check and cam2 (REAR) captures the container
    f_cam2 = replace(f, camera_id=cam2.id)
    measure(r, cam2, f_cam2, 15, [('truck', (30, 20, 610, 460)), ('container_number', (110, 80, 190, 140))])
    drain(r)
    r._finalize_passages()

    # Same event must now have both camera proofs, cloud_synced reset to False, and re-enqueued for PUT /images override
    events_after = repository.events()['items']
    assert len(events_after) == 1
    updated_detail = repository.event(ev_id)
    assert {s['camera_id'] for s in updated_detail['snapshots']} == {'gate-out-2', 'gate-out-1'}
    assert updated_detail['cloud_synced'] is False
    assert updated_detail['cloud_event_id'] == '0e7d4fac-5f22-48a4-be37-0bc58a948eff'
    assert r.dispatcher.enqueue.call_count == 2


def test_back_to_back_out_gate_containers_not_skipped(settings, repository, snapshots, camera, make_job):
    """When a second container immediately follows a first container at the OUT gate (even reusing track_id or
    arriving right after early finalization), both containers must be recorded as distinct events."""
    settings.shared_visit_folders = True
    settings.container_roi_trigger = True
    settings.ocr_confirmations = 1
    settings.minimum_number_proof_cameras = 1
    cam_out = camera.model_copy(update={'id': 'gate-out-1', 'role': 'REAR', 'gate_id': 'lane-out', 'direction_by_heading': 'EXIT'})
    repository.sync_cameras([cam_out])
    r, f = setup(settings, repository, snapshots, cam_out, make_job)
    f = replace(f, camera_id=cam_out.id)
    r.cameras = [cam_out]

    # Container 1 (CSQU3054383, bbox x < 200 in drain helper) passes at t=0
    measure(r, cam_out, f, 0, [('truck', (20, 20, 600, 460)), ('container_number', (80, 80, 160, 140))])
    drain(r)
    r._finalize_passages()
    assert len(repository.events()['items']) == 1

    # Container 2 (MSCU6639870, bbox x >= 200 in drain helper) arrives immediately at t=3 with distinct appearance
    f2 = replace(f, image=np.random.default_rng(7).integers(0, 255, (480, 640, 3), dtype=np.uint8))
    measure(r, cam_out, f2, 3, [('truck', (25, 25, 605, 465)), ('container_number', (320, 80, 420, 140))])
    drain(r)
    r._finalize_passages()

    items = repository.events()['items']
    assert len(items) == 2
    assert {e['container_number'] for e in items} == {'CSQU3054383', 'MSCU6639870'}


def test_early_finalization_with_unconfirmed_secondary_text_while_stopped_at_gate(settings, repository, snapshots, camera, make_job):
    """When a truck is stopped at Gate OUT with a confirmed container number AND unconfirmed secondary
    text (e.g. slogan, tare weight, or companion camera text), it must finalize immediately while waiting
    without requiring departure or stalling on unconfirmed text regions."""
    settings.shared_visit_folders = True
    settings.container_roi_trigger = True
    settings.ocr_confirmations = 1
    settings.minimum_number_proof_cameras = 1
    cam_out = camera.model_copy(update={'id': 'gate-out-1', 'role': 'REAR', 'gate_id': 'lane-out', 'direction_by_heading': 'EXIT'})
    repository.sync_cameras([cam_out])
    r, f = setup(settings, repository, snapshots, cam_out, make_job)
    f = replace(f, camera_id=cam_out.id)
    r.cameras = [cam_out]

    # Truck arrives at Gate OUT with confirmed number (x < 200) AND unconfirmed secondary text (x >= 200)
    measure(r, cam_out, f, 0, [
        ('truck', (20, 20, 600, 460)),
        ('container_number', (80, 80, 160, 140)),
        ('container_number', (320, 300, 420, 350))
    ])
    while not r.queue.empty():
        job = r.queue.get_nowait()
        key = r.ocr_store.save(job, job.frame.image[20:80, 20:100])
        if job.detection.bbox[0] < 200:
            n = 'CSQU3054383'
            r.ocr_store.save_validation(key, Validation(n, n, 0.95, True, True, 'VALID'))
        else:
            r.ocr_store.save_validation(key, Validation('CAUTION 2.6M', 'CAUTION 2.6M', 0.35, False, False, 'INVALID'))
        r._process_saved_ocr(key)
        r.queue.task_done()

    # The truck is STILL inside gate, passage is NOT closed
    active_passages = list(r.passages.passages.values())
    assert any(not p.closed for p in active_passages)

    r._finalize_passages()

    # Must be finalized, confirmed, and enqueued immediately while vehicle is stopped
    events = repository.events()['items']
    assert len(events) == 1
    assert events[0]['container_number'] == 'CSQU3054383'
    assert events[0]['status'] == 'CONFIRMED'
    assert r.dispatcher.enqueue.call_count == 1


def test_pseudo_twin_merge_for_same_container_ocr_variants(settings, repository, snapshots, camera, make_job):
    """Two OCR variants of the same single container (e.g. ONEU3378921 and ONEU4337892)
    must merge into one single container event with the higher-quality read, not a twin 20ft."""
    from app.events.passages import Passage, Region
    from datetime import datetime, timezone
    from app.domain import Validation, Detection
    from dataclasses import replace

    now = datetime.now(timezone.utc)
    passage = Passage(
        id='pass-twin-merge', camera_id=camera.id, source_id='src-1',
        gate_id=camera.gate_id, lane_id=camera.gate_id, direction='ENTRY',
        started=now, last_seen=now, track_id=1, bbox=(10, 10, 500, 300),
        final_evidence=True
    )
    job1 = make_job(sequence=1, camera_id=camera.id, timestamp=now)
    job2 = make_job(sequence=2, camera_id=camera.id, timestamp=now)
    v1 = Validation('ONEU3378921', 'ONEU3378921', 0.90, True, True, 'VALID')
    v2 = Validation('ONEU4337892', 'ONEU4337892', 0.98, True, True, 'VALID')

    r1 = Region(id='r1', bbox=(0.1, 0.1, 0.3, 0.3))
    r1.votes['ONEU3378921'] = 1
    r1.reads['ONEU3378921'] = (v1, job1)
    r1.source_proofs['ONEU3378921'] = {camera.id: (v1, job1)}

    r2 = Region(id='r2', bbox=(0.2, 0.1, 0.4, 0.3))
    r2.votes['ONEU4337892'] = 1
    r2.reads['ONEU4337892'] = (v2, job2)
    r2.source_proofs['ONEU4337892'] = {camera.id: (v2, job2)}

    passage.regions = [r1, r2]

    ev_id = repository.save_passage(passage, snapshots, confirmations=1)
    ev = repository.event(ev_id)

    assert ev['is_twin'] is False
    assert ev['container_number'] == 'ONEU4337892'
    assert ev['container_number_2'] is None


def test_companion_camera_attaches_only_with_detected_frames_and_not_blind_context(settings, repository, snapshots, camera, make_job):
    """Cameras must ONLY attach detection frames if they actually observed the vehicle (_frame.png).
    Cameras with only ambient _context_*.jpg or disabled cameras must NEVER attach snapshots."""
    from app.events.passages import Passage, Region
    from datetime import datetime, timezone
    from app.domain import Validation
    import cv2, numpy as np

    now = datetime.now(timezone.utc)
    cam_in1 = camera.model_copy(update={'id': 'gate-in-1', 'gate_id': 'lane-in', 'enabled': True})
    cam_in2 = camera.model_copy(update={'id': 'gate-in-2', 'gate_id': 'lane-in', 'enabled': True})
    cam_in3 = camera.model_copy(update={'id': 'gate-in-3', 'gate_id': 'lane-in', 'enabled': False})  # Disabled
    repository.sync_cameras([cam_in1, cam_in2, cam_in3])

    passage = Passage(
        id='pass-comp-test', camera_id='gate-in-2', source_id='src-2',
        gate_id='lane-in', lane_id='lane-in', direction='ENTRY',
        started=now, last_seen=now, track_id=1, bbox=(10, 10, 500, 300),
        final_evidence=True
    )
    job2 = make_job(sequence=1, camera_id='gate-in-2', timestamp=now)
    v2 = Validation('MSKU1234567', 'MSKU1234567', 0.95, True, True, 'VALID')
    r = Region(id='r1', bbox=(0.1, 0.1, 0.3, 0.3))
    r.votes['MSKU1234567'] = 1
    r.reads['MSKU1234567'] = (v2, job2)
    r.source_proofs['MSKU1234567'] = {'gate-in-2': (v2, job2)}
    passage.regions = [r]

    # Set up visit directory with:
    # 1. gate-in-1 has a detection frame _frame.png (it detected the vehicle)
    # 2. gate-in-3 (disabled / office) has only _context_0.jpg
    cat = snapshots.category_for('ENTRY')
    proof_dir = snapshots.root / cat / passage.id
    proof_dir.mkdir(parents=True, exist_ok=True)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    ok, enc = cv2.imencode('.png', dummy_img)
    (proof_dir / 'gate-in-1_10_frame.png').write_bytes(enc.tobytes())
    (proof_dir / 'gate-in-3_context_0.jpg').write_bytes(enc.tobytes())

    ev_id = repository.save_passage(passage, snapshots, confirmations=1)
    ev = repository.event(ev_id)

    snap_cams = {s['camera_id'] for s in ev['snapshots']}
    # gate-in-2 (primary proof) and gate-in-1 (observed companion detection frame) must be present
    assert 'gate-in-2' in snap_cams
    assert 'gate-in-1' in snap_cams
    # gate-in-3 (disabled and only had _context_0.jpg) must NOT be present
    assert 'gate-in-3' not in snap_cams





