import { Component, effect, input, output, signal, untracked } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { Camera, LaneCalibration } from './gate-api.service';

@Component({
  selector: 'gate-roi-editor', standalone: true, imports: [DecimalPipe],
  template: `
    <div class="roi-editor" aria-label="Camera detection area and proof position">
      <div class="roi-toolbar">
        <div class="roi-title"><span class="roi-icon">1</span><div>
          <strong>Where should this camera look for passing containers?</strong>
          <small>Draw the green area over the truck path. Leave parked container stacks and unrelated roads outside it.</small>
        </div></div>
        <div class="roi-context"><span class="calibration-pill">{{ camera().name }}</span><span class="calibration-pill">{{ camera().direction === 'ENTRY' ? 'Gate IN' : 'Gate OUT' }}</span></div>
      </div>
      <div class="roi-actions">
        <button type="button" [class.selected]="tool() === 'draw'" (click)="laneIndex.set(-1); tool.set('draw')">Draw detection area</button>
        <button type="button" [class.selected]="tool() === 'adjust'" (click)="laneIndex.set(-1); tool.set('adjust')">Move / resize area</button>
        <span>{{ tool() === 'proof' ? 'Click on the image to place the yellow proof line.' : 'Drag across the image to draw, or use Move / resize.' }}</span>
      </div>
      <div class="roi-canvas" [style.aspect-ratio]="aspect()" (pointerdown)="start($event)" (pointermove)="move($event)" (pointerup)="end($event)" (pointercancel)="cancel()">
        <img [src]="'/api/cameras/' + camera().id + '/frame?calibration=' + refresh()" [alt]="camera().name + ' calibration frame'" draggable="false" (load)="frameLoaded($event)" (error)="ready.set(false)">
        @if (ready()) {
          <svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
            <path [attr.d]="shade()" fill-rule="evenodd" fill="rgba(0,0,0,.5)"/>
            <rect [attr.x]="roi()[0]" [attr.y]="roi()[1]" [attr.width]="roi()[2]-roi()[0]" [attr.height]="roi()[3]-roi()[1]" class="roi-box"/>
            @for (corner of corners; track corner[0]) {
              <circle [attr.cx]="roi()[corner[1]]" [attr.cy]="roi()[corner[2]]" r=".008" fill="white" stroke="#22c55e" stroke-width=".003"/>
            }
            @for (lane of config().lanes || []; track $index) {
              <polygon [attr.points]="lanePoints(lane.polygon)" fill="#38bdf833" stroke="#38bdf8" stroke-width=".003"/>
            }
            @if (config().proof_capture_axis === 'x') {
              <line [attr.x1]="config().proof_capture_position" y1="0" [attr.x2]="config().proof_capture_position" y2="1" stroke="#fbbf24" stroke-width=".006"/>
            }
            @if (config().proof_capture_axis === 'y') {
              <line x1="0" [attr.y1]="config().proof_capture_position" x2="1" [attr.y2]="config().proof_capture_position" stroke="#fbbf24" stroke-width=".006"/>
            }
          </svg>
        } @else {
          <div class="roi-unavailable">Connect this camera or load a video to draw its area.</div>
        }
      </div>
      <div class="roi-readout"><span>Green area: detection and OCR</span><span>{{ dirty() ? 'Unsaved changes' : 'Saved settings' }}</span></div>
      <div class="roi-clearance">
        <strong>2. Where should the client proof be captured?</strong>
        <p>Set one yellow reference line for this gate. When the detected truck/container centre crosses it, all assigned cameras use that capture time. OCR continues using other clearer frames.</p>
        <label>Camera's proof role
          <select [value]="config().proof_capture_axis || ''" (change)="setProofAxis($event)">
            <option value="">Follow the gate's reference camera</option>
            <option value="x">Reference camera: vertical proof line</option>
            <option value="y">Reference camera: horizontal proof line</option>
          </select>
        </label>
        @if (config().proof_capture_axis) {
          <div class="roi-boundary-controls">
            <button type="button" [class.selected]="tool() === 'proof'" (click)="laneIndex.set(-1);tool.set('proof')">Place yellow line on the image</button>
            <label>Proof line position: {{ (config().proof_capture_position || .5)*100 | number:'1.0-0' }}%
              <input type="range" min=".02" max=".98" step=".01" [value]="config().proof_capture_position || .5" (input)="setProofPosition($event)">
            </label>
            <label>Truck movement
              <select [value]="config().proof_capture_positive === false ? 'negative' : 'positive'" (change)="setProofTravel($event)">
                <option value="positive">Right / down across the line</option><option value="negative">Left / up across the line</option>
              </select>
            </label>
          </div>
        } @else {
          <p>This camera follows the reference selected on another camera. If none is selected for this gate, proof remains available for review with a calibration warning.</p>
        }
        <p>No separate entry and exit lines are needed. The application tracks the truck leaving its area and waits for clearance automatically.</p>
      </div>
      <details class="roi-clearance">
        <summary>Optional: two truck lanes in this camera</summary>
        <p>Use this only when two trucks can pass side by side. Add a lane, then click its corners on the image. Keep each lane polygon convex.</p>
        @for (lane of config().lanes || []; track $index) {
          <button type="button" (click)="laneIndex.set($index)">{{ lane.id }}</button>
        }
        <button type="button" [disabled]="(config().lanes?.length || 0) >= 2" (click)="addLane()">Add lane</button>
        @if (laneIndex() >= 0) {
          <button type="button" (click)="clearLane()">Redraw selected lane</button>
          <button type="button" (click)="removeLane()">Remove selected lane</button>
          <button type="button" (click)="laneIndex.set(-1)">Finish lane</button>
        }
      </details>
      <div class="roi-controls">
        <button type="button" class="roi-reset" [disabled]="!dirty()" (click)="restore()">Discard changes</button>
        <button type="button" class="primary" [disabled]="!dirty() || !ready() || drawing" (click)="submit()">Save camera area & proof position</button>
      </div>
    </div>`
})
export class RoiEditorComponent {
  camera = input.required<Camera>(); calibration = input.required<LaneCalibration>(); refresh = input(0); save = output<LaneCalibration>();
  roi = signal<number[]>([0, 0, 1, 1]); config = signal<LaneCalibration>({camera_id:'', occupancy_roi:[0,0,1,1], entry_line:.1, exit_line:.85, gate_exit_axis:null, gate_exit_positive:true, version:1});
  dirty = signal(false); ready = signal(false); aspect = signal(16/9); tool = signal<'draw'|'adjust'|'proof'>('draw');
  laneIndex = signal(-1);
  lanePoints(points:number[][]){return points.map(p=>p.join(',')).join(' ');}
  addLane(){this.tool.set('adjust');this.config.update(c=>({...c,lanes:[...(c.lanes || []),{id:'lane-'+((c.lanes?.length || 0)+1),polygon:[]}]}));this.laneIndex.set((this.config().lanes?.length || 1)-1);this.dirty.set(true);}
  clearLane(){this.config.update(c=>({...c,lanes:c.lanes?.map((l,i)=>i===this.laneIndex()?{...l,polygon:[]}:l)}));this.dirty.set(true);}
  removeLane(){this.config.update(c=>({...c,lanes:c.lanes?.filter((_,i)=>i!==this.laneIndex())}));this.laneIndex.set(-1);this.dirty.set(true);}
  renameLane(e:Event){const id=(e.target as HTMLInputElement).value.trim();this.config.update(c=>({...c,lanes:c.lanes?.map((l,i)=>i===this.laneIndex()?{...l,id}:l)}));this.dirty.set(true);}
  corners = [['tl',0,1],['tr',2,1],['bl',0,3],['br',2,3]] as const; drawing=false; private cameraId=''; private anchor=[0,0]; private before=[0,0,1,1]; private mode='draw';
  constructor() { effect(() => { const value=this.calibration(), id=this.camera().id; untracked(() => { const changed=id !== this.cameraId; if (changed) { this.cameraId=id; this.ready.set(false); this.drawing=false; } if (!this.dirty() || changed) { this.config.set({...value}); this.roi.set([...value.occupancy_roi]); this.dirty.set(false); } }); }); }
  frameLoaded(e: Event) { const i=e.target as HTMLImageElement; if(i.naturalWidth&&i.naturalHeight){this.aspect.set(i.naturalWidth/i.naturalHeight);this.ready.set(true);} }
  shade(){const[a,b,c,d]=this.roi();return `M0 0H1V1H0Z M${a} ${b}V${d}H${c}V${b}Z`;}
  start(e:PointerEvent){if(!this.ready()||e.button!==0)return;const p=this.point(e);if(this.tool()==='proof'){this.config.update(c=>({...c,proof_capture_position:Math.max(.02,Math.min(.98,p[c.proof_capture_axis==='x'?0:1]))}));this.dirty.set(true);return;}if(this.laneIndex()>=0){this.config.update(c=>({...c,lanes:c.lanes?.map((l,i)=>i===this.laneIndex()&&l.polygon.length<12?{...l,polygon:[...l.polygon,p]}:l)}));this.dirty.set(true);return;}const r=(e.currentTarget as HTMLElement).getBoundingClientRect();this.before=[...this.roi()];this.anchor=p;this.mode='draw';if(this.tool()==='adjust'){const q=this.corners.find(c=>Math.abs(p[0]-this.before[c[1]])*r.width<18&&Math.abs(p[1]-this.before[c[2]])*r.height<18);if(q)this.mode=q[0];else if(p[0]>=this.before[0]&&p[0]<=this.before[2]&&p[1]>=this.before[1]&&p[1]<=this.before[3])this.mode='move';else return;}this.drawing=true;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);}
  move(e:PointerEvent){if(!this.drawing)return;const[x,y]=this.point(e),b=this.before;if(this.mode==='draw')this.roi.set([Math.min(this.anchor[0],x),Math.min(this.anchor[1],y),Math.max(this.anchor[0],x),Math.max(this.anchor[1],y)]);else if(this.mode==='move'){const dx=Math.max(-b[0],Math.min(1-b[2],x-this.anchor[0])),dy=Math.max(-b[1],Math.min(1-b[3],y-this.anchor[1]));this.roi.set([b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy]);}else{const n=[...b];if(this.mode.endsWith('l'))n[0]=Math.min(x,b[2]-.02);else n[2]=Math.max(x,b[0]+.02);if(this.mode.startsWith('t'))n[1]=Math.min(y,b[3]-.02);else n[3]=Math.max(y,b[1]+.02);this.roi.set(n);}}
  end(e:PointerEvent){if(!this.drawing)return;this.move(e);this.drawing=false;const b=this.roi();if(b[2]-b[0]<.02||b[3]-b[1]<.02)this.roi.set(this.before);else if(JSON.stringify(b)!==JSON.stringify(this.before)){this.dirty.set(true);this.tool.set('adjust');}}
  cancel(){if(this.drawing)this.roi.set(this.before);this.drawing=false;} fullFrame(){this.roi.set([0,0,1,1]);this.dirty.set(true);} restore(){const c=this.calibration();this.config.set({...c});this.roi.set([...c.occupancy_roi]);this.dirty.set(false);}
  setProofAxis(e:Event){const value=(e.target as HTMLSelectElement).value;this.config.update(c=>({...c,proof_capture_axis:value ? value as 'x'|'y' : null,proof_capture_position:c.proof_capture_position || .5}));this.dirty.set(true);}
  setProofPosition(e:Event){this.config.update(c=>({...c,proof_capture_position:Number((e.target as HTMLInputElement).value)}));this.dirty.set(true);}
  setProofTravel(e:Event){this.config.update(c=>({...c,proof_capture_positive:(e.target as HTMLSelectElement).value==='positive'}));this.dirty.set(true);}
  submit(){this.save.emit({...this.config(),gate_exit_axis:null,occupancy_roi:[...this.roi()]});this.dirty.set(false);}
  private point(e:PointerEvent){const r=(e.currentTarget as HTMLElement).getBoundingClientRect();return[Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))];}
}

