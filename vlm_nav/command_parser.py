"""LLM prompt + response grammar for the robot brain (pure python, no ROS / torch).

Grammar (the command goes on the first line, an explanation may follow):
    GOAL:<object name>      - go to a named object; the *code* looks up its coordinates
    RELATIVE:(dx,dy)        - move relative to the robot (dx forward, dy left, metres)
    MOVE:<direction> <m>    - same, with a direction word (forward/back/left/right): easier for small models
                              than signed numbers, and parsed into the same RELATIVE command
    STOP                    - only when STOP is the first word of the reply (models often add ' - explanation')
    anything else           - plain text answer
"""
import math
import re
from dataclasses import dataclass
from typing import Optional

_NUM = r'[+-]?(?:\d+\.?\d*|\.\d+)'
_GOAL_RE = re.compile(r'^\W*GOAL\s*:\s*(?P<rest>.+)$', re.IGNORECASE)
_REL_RE = re.compile(rf'^\W*RELATIVE\s*:\s*\(?\s*(?P<dx>{_NUM})\s*,\s*(?P<dy>{_NUM})\s*\)?', re.IGNORECASE)
_MOVE_RE = re.compile(rf'^\W*MOVE\s*:\s*(?P<dir>forwards?|ahead|backwards?|back|left|right)\s+(?P<m>{_NUM})', re.IGNORECASE)
_MOVE_DIRS = {'forward': (1, 0), 'forwards': (1, 0), 'ahead': (1, 0), 'back': (-1, 0), 'backward': (-1, 0),
              'backwards': (-1, 0), 'left': (0, 1), 'right': (0, -1)}
_STOP_RE = re.compile(r'^\W*STOP\b', re.IGNORECASE)
_NAME_TRIM = ' \t"\'`(<[*'


@dataclass
class Command:
    kind: str                       # 'GOAL' | 'RELATIVE' | 'STOP' | 'TEXT'
    target: Optional[str] = None    # GOAL: object id or label as written by the model
    dx: float = 0.0
    dy: float = 0.0


def _match_target(rest, known_names):
    """Pick the object name at the start of `rest`, preferring the longest known name."""
    rest = rest.strip(_NAME_TRIM)
    low = rest.lower()
    for name in sorted(known_names, key=len, reverse=True):
        if low.startswith(name) and (len(low) == len(name) or not (low[len(name)].isalnum() or low[len(name)] == '_')):
            return name
    m = re.match(r'[\w-]+', rest)
    return m.group(0).lower() if m else ''


def parse_response(response, known_names=()):
    """Turn the model's reply into a Command.

    GOAL and RELATIVE are checked before STOP, and STOP only counts when it is the
    first word of the first non-empty line ('STOP', 'STOP - stopping now'), so
    'GOAL:shelf - I will stop there' or a sentence that merely contains the word
    'stop' can never cancel motion.
    """
    known = [n.lower() for n in known_names]
    lines = [ln for ln in response.splitlines() if ln.strip()]

    rel = None
    for ln in lines:
        m = _GOAL_RE.match(ln)
        if m:
            return Command('GOAL', target=_match_target(m.group('rest'), known))
        if rel is None:
            m = _REL_RE.match(ln)
            if m:
                rel = Command('RELATIVE', dx=float(m.group('dx')), dy=float(m.group('dy')))
            else:
                m = _MOVE_RE.match(ln)
                if m:
                    ux, uy = _MOVE_DIRS[m.group('dir').lower()]
                    dist = abs(float(m.group('m')))
                    rel = Command('RELATIVE', dx=ux * dist, dy=uy * dist)
    if rel is not None:
        return rel
    if lines and _STOP_RE.match(lines[0]):
        return Command('STOP')
    return Command('TEXT')


_COMMAND_START_RE = re.compile(r'^\W*(GOAL|RELATIVE|MOVE|STOP)\b', re.IGNORECASE)


def first_command_line(response):
    """If the reply starts with a command, return just that line.

    Small models sometimes append invented extra lines after a command (e.g. a stray 'RELATIVE:(0,0)');
    only the command line is meant to be acted on and shown. Plain-text replies are returned unchanged.
    """
    lines = [ln for ln in response.splitlines() if ln.strip()]
    if lines and _COMMAND_START_RE.match(lines[0]):
        return lines[0].strip()
    return response.strip()


def _objects_block(graph, robot_pose):
    by_label = {}
    for o in graph.objects:
        by_label.setdefault(o.label, []).append(o)
    lines = []
    for label, objs in sorted(by_label.items()):
        if robot_pose is None:
            items = ', '.join(o.id for o in objs)
        else:
            items = ', '.join(f'{o.id} ({math.hypot(o.x - robot_pose[0], o.y - robot_pose[1]):.0f} m)' for o in objs)
        lines.append(f'{label}: {items}')
    return '\n'.join(lines)


def build_system_prompt(graph, robot_pose=None):
    """System prompt for a small VLM. Objects are listed by *name*: the model never handles goal coordinates.

    Worked examples matter more than rules for a 3B model (without them it answers 'go forward' with a GOAL and
    parrots the instructions), so the prompt ends with examples built from the objects that actually exist.
    """
    if robot_pose is None:
        pose_line = "The robot's position is unknown right now."
        where = "I don't know my position right now."
    else:
        x, y, yaw = robot_pose
        pose_line = f'Robot pose in the map: x={x:.2f}, y={y:.2f}, heading {math.degrees(yaw):.0f} degrees.'
        where = f'I am at x={x:.2f}, y={y:.2f}, facing {math.degrees(yaw):.0f} degrees.'
    have = graph is not None and len(graph) > 0
    ctx = f'You control a mobile robot with a front camera. {pose_line}\n\n'
    listing = ' (id, distance from the robot):\n' + _objects_block(graph, robot_pose) if have else ': none'
    ctx += f'Known objects{listing}\n\n'
    ctx += ('Reply with ONE line:\n'
            'GOAL:<object id or class>  - go to a listed object (use the class, e.g. forklift, to mean the nearest one)\n'
            'MOVE:<forward|back|left|right> <meters>  - move relative to the robot\n'
            'STOP  - stop moving\n'
            'For questions, answer in plain text. If the user asks for an object that is not listed, say you do '
            'not know it. If an image is provided and the user asks what you see, describe it briefly in 2-3 lines.\n\n')
    examples = []
    if have:
        labels = graph.labels()
        first = next((o for o in graph.objects if o.label == labels[0]), None)
        second = next((o for o in graph.objects if o.label == labels[-1]), first)
        examples += [(f'go to the {first.label}', f'GOAL:{first.label}'), (f'take me to {second.id}', f'GOAL:{second.id}')]
    examples += [('go forward 2 meters', 'MOVE:forward 2'), ('move back 1 meter', 'MOVE:back 1'),
                 ('go left 1.5 meters', 'MOVE:left 1.5'), ('go right 3 meters', 'MOVE:right 3'), ('stop', 'STOP'),
                 ('go to the spaceship', "I don't know an object called spaceship."), ('where are you?', where)]
    ctx += 'Examples:\n' + '\n'.join(f'User: {u}\nReply: {a}' for u, a in examples) + '\n'
    return ctx


_QUESTION_RE = re.compile(r'^\W*(?:which|what|where|who|why|when|how)\b(?!\s+about)', re.IGNORECASE)


def is_question(user_input):
    """True for messages that start with a question word ('which forklift is closest?', 'what do you see?').

    Such a message is never an order, so the brain answers it in words with an answer-only prompt and does not
    act on any command the model might still emit (a 3B model sometimes answers a question with a GOAL).
    """
    return bool(_QUESTION_RE.match(user_input))


def build_question_prompt(graph, robot_pose=None):
    """Answer-only system prompt: same facts as build_system_prompt, but no commands are offered."""
    if robot_pose is None:
        pose_line = "The robot's position is unknown right now."
    else:
        x, y, yaw = robot_pose
        pose_line = f'Robot pose in the map: x={x:.2f}, y={y:.2f}, heading {math.degrees(yaw):.0f} degrees.'
    have = graph is not None and len(graph) > 0
    listing = ' (id, distance from the robot):\n' + _objects_block(graph, robot_pose) if have else ': none'
    return (f'You are the assistant of a mobile robot with a front camera. {pose_line}\n\n'
            f'Known objects{listing}\n\n'
            "Answer the user's question briefly in plain text, using only the information above (and the camera image "
            'if one is provided). Do not output commands.')


VISION_TRIGGERS = ['see', 'look', 'camera', 'view', 'around you', 'in front']


def needs_vision(user_input):
    text = user_input.lower()
    return any(trigger in text for trigger in VISION_TRIGGERS)
