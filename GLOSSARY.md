# Cutter

Cutter is a local tool that makes an editable rough cut of one English video from one speaker's sources, recorded long-form on one camera.

## Language

**Cutter**:
The tool that produces a rough cut from a project's English sources.
_Avoid_: Cutting tool

**Project**:
One output video, kept in its own folder: the rough cut made from that folder's sources.
_Avoid_: Session, recording day

**Source**:
One raw media file in a project.
_Avoid_: Clip, input video

**Take**:
One attempt at a passage of speech.

**Retake**:
A later take that repeats an aborted failed take. The later take is the one that stays.
_Avoid_: Restart

**Failed take**:
An earlier take that was aborted before the sentence ended, and that a retake replaces.
_Avoid_: a finished sentence repeated on purpose

**Decision**:
The judgment on one candidate repeat: drop the failed take, or keep the earlier span for a person to review.

**False cut**:
Speech the finished edit keeps that the rough cut removed.
_Avoid_: Overcut

**Missed retake**:
A failed take that the rough cut still plays.

**Range**:
A continuous span of one source that plays in the rough cut.

**Rough cut**:
The editable sequence of kept ranges.

**Rejects**:
The sequence of failed takes that were dropped from the rough cut.

**CHECK**:
A marker, resolved in Final Cut, that a person should review a decision before trusting the cut.

**Gold**:
A finished human edit of a project, the reference for which words should have been kept.

**Profile**:
The named thresholds for one recording shape. `long` is the horizontal cut.
`short` is the vertical cut. When `--profile` is omitted, the frame decides.
