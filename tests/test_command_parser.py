import pytest

from vlm_nav.command_parser import (build_question_prompt, build_system_prompt, first_command_line, is_question,
                                    mentioned_objects, needs_vision, parse_response, reconcile_goal)
from vlm_nav.scene_graph import SceneGraph, SceneObject

NAMES = ['forklift', 'forklift_1', 'shelf', 'shelf_1', 'box', 'box_2']


def test_goal_by_name():
    c = parse_response('GOAL:forklift_1\nGoing to the forklift.', NAMES)
    assert (c.kind, c.target) == ('GOAL', 'forklift_1')


def test_goal_wins_over_the_word_stop_in_the_explanation():
    # The reported bug: STOP was matched anywhere in the text before GOAL was even considered.
    c = parse_response("GOAL:forklift - I'll stop at the shelf", NAMES)
    assert (c.kind, c.target) == ('GOAL', 'forklift')
    c = parse_response("GOAL:shelf_1\nI will stop in front of it.", NAMES)
    assert (c.kind, c.target) == ('GOAL', 'shelf_1')


def test_legacy_numeric_goal_never_becomes_stop_or_a_valid_target():
    c = parse_response("GOAL:(3,4) - I'll stop at the shelf", NAMES)
    assert c.kind == 'GOAL' and c.target not in NAMES


def test_goal_priority_over_bare_stop_line():
    assert parse_response('STOP\nGOAL:shelf', NAMES).kind == 'GOAL'


@pytest.mark.parametrize('reply', ['STOP', 'STOP\nStopping the robot.', '**STOP**', 'stop.', '  STOP  \n\nok',
                                   'STOP - The robot has stopped its current movement.', 'Stop! stopping now'])
def test_bare_stop_line_stops(reply):
    assert parse_response(reply, NAMES).kind == 'STOP'


@pytest.mark.parametrize('reply', [
    "I will stop at the shelf.", "Sure, I'll STOP when I get there", 'The sign says STOP.',
    'Please do not stop', 'Robot: nothing to do', 'Stopping at the shelf now.', 'I would stop at the shelf.',
])
def test_stop_inside_prose_is_plain_text(reply):
    assert parse_response(reply, NAMES).kind == 'TEXT'


@pytest.mark.parametrize('reply,dx,dy', [
    ('RELATIVE:(1.5,-0.5)\nMoving.', 1.5, -0.5),
    ('RELATIVE:( -2 , 0 )', -2.0, 0.0),
    ('RELATIVE:(.5,1)', 0.5, 1.0),
    ('relative:(3,0)', 3.0, 0.0),
])
def test_relative(reply, dx, dy):
    c = parse_response(reply, NAMES)
    assert (c.kind, c.dx, c.dy) == ('RELATIVE', dx, dy)


@pytest.mark.parametrize('reply,dx,dy', [
    ('MOVE:forward 1', 1.0, 0.0), ('MOVE:back 2.5', -2.5, 0.0), ('move: left 1.5 m', 0.0, 1.5),
    ('MOVE:right 3\nOn my way', 0.0, -3.0), ('MOVE: backward 1', -1.0, 0.0), ('MOVE:left -2', 0.0, 2.0),
])
def test_move_with_direction_words(reply, dx, dy):
    c = parse_response(reply, NAMES)
    assert (c.kind, c.dx, c.dy) == ('RELATIVE', dx, dy)


def test_goal_name_resolution_respects_word_boundaries_and_case():
    assert parse_response('GOAL: Forklift_1', NAMES).target == 'forklift_1'
    assert parse_response('GOAL:"shelf"', NAMES).target == 'shelf'
    assert parse_response('GOAL:box_2 now', NAMES).target == 'box_2'
    assert parse_response('GOAL:boxes', NAMES).target == 'boxes'      # not a known name -> stays unresolved


def test_plain_text():
    assert parse_response('I see a warehouse aisle with shelves.', NAMES).kind == 'TEXT'
    assert parse_response('', NAMES).kind == 'TEXT'


def _graph():
    return SceneGraph([SceneObject('forklift_1', 'forklift', 12.34, -5.67, 0.1, 4, 0.7),
                       SceneObject('shelf_1', 'shelf', 3.21, 8.76, 0.2, 3, 0.6)])


def test_prompt_lists_names_but_never_object_coordinates():
    p = build_system_prompt(_graph(), (0.0, 0.0, 0.0))
    assert 'forklift_1' in p and 'shelf_1' in p
    for leaked in ('12.34', '-5.67', '3.21', '8.76'):
        assert leaked not in p
    assert 'GOAL:<object id or class>' in p and 'MOVE:<forward|back|left|right>' in p


def test_prompt_handles_unknown_pose_and_empty_graph():
    assert 'position is unknown' in build_system_prompt(_graph(), None)
    assert 'Known objects: none' in build_system_prompt(SceneGraph([]), (0, 0, 0))


def test_needs_vision():
    assert needs_vision('What do you see?')
    assert not needs_vision('go to the forklift')


def test_prompt_examples_use_real_objects_and_the_real_pose():
    p = build_system_prompt(_graph(), (1.0, 2.0, 0.0))
    assert 'User: go to the forklift\nReply: GOAL:forklift' in p and 'GOAL:shelf_1' in p
    assert 'Reply: I am at x=1.00, y=2.00, facing 0 degrees.' in p
    assert 'forklift: forklift_1 (' in p


def test_first_command_line_drops_invented_trailing_lines():
    assert first_command_line('GOAL:forklift\nRELATIVE:(0.0,12.0)') == 'GOAL:forklift'
    assert first_command_line('STOP - stopping\nmore') == 'STOP - stopping'
    assert first_command_line('I see a forklift.\nIt is yellow.') == 'I see a forklift.\nIt is yellow.'


@pytest.mark.parametrize('text,expected', [
    ('which forklift is closest to you?', True), ('What do you see?', True), ('where is the shelf', True),
    ('how far is the door?', True), ('how many forklifts are there', True), ('who are you', True),
    ('go to the forklift', False), ('how about going to the shelf', False), ('please go forward', False),
    ('can you go to the door?', False), ('stop', False), ('  Which one is nearer?', True),
])
def test_is_question(text, expected):
    assert is_question(text) is expected


def test_question_prompt_offers_no_commands_but_keeps_the_facts():
    p = build_question_prompt(_graph(), (0.0, 0.0, 0.0))
    assert 'forklift_1' in p and 'x=0.00' in p
    assert 'GOAL' not in p and 'MOVE' not in p and 'Do not output commands' in p


# ---------------------------------------------------------------- names with spaces / underscores / articles

HOSPITAL_NAMES = ['vending machine', 'vending machine_1', 'trash can', 'trash can_2', 'bed', 'bed_1']


@pytest.mark.parametrize('reply,target', [
    ('GOAL:vending_machine_1', 'vending machine_1'),
    ('GOAL: the trash_can_2', 'trash can_2'),
    ('GOAL:Trash Can 2 - going there', 'trash can_2'),
    ('GOAL:the bed', 'bed'),
    ('goal: nearest vending machine', 'vending machine'),
    ('GOAL:bed_1.', 'bed_1'),
])
def test_goal_targets_tolerate_underscores_articles_and_case(reply, target):
    assert parse_response(reply, HOSPITAL_NAMES).target == target


@pytest.mark.parametrize('text,expected', [
    ('what do you see?', True), ('look around', True), ('show me the camera view', True),
    ('go to the seat', False), ('interview the nurse', False), ('go to the bed', False), ('what is in front of you', True),
])
def test_vision_triggers_are_whole_words(text, expected):
    assert needs_vision(text) is expected


def test_prompt_marks_groups_and_caps_long_lists():
    many = [SceneObject(f'door_{i}', 'door', float(i), 0.0, 1.0, 5, 0.5) for i in range(1, 12)]
    pile = SceneObject('box_1', 'box', 20.0, 0.0, 0.5, 9, 0.5, kind='group', members=6, size=4.2)
    prompt = build_system_prompt(SceneGraph(many + [pile]), (0.0, 0.0, 0.0))
    assert 'box_1 (group of 6, 4 m wide, 20 m)' in prompt
    assert 'door_1 (1 m)' in prompt and '+5 more' in prompt and 'door_11' not in prompt


# ---------------------------------------------------------------- the user's words beat a small model's arbitrary pick

def _carts():
    return SceneGraph([SceneObject('cart_1', 'cart', 20.0, 0.0, 0.5, 5, 0.5),
                       SceneObject('cart_3', 'cart', -4.0, 0.0, 0.5, 5, 0.5),
                       SceneObject('vending machine_1', 'vending machine', 9.0, 9.0, 1.0, 5, 0.5),
                       SceneObject('vending machine_2', 'vending machine', 9.0, -9.0, 1.0, 5, 0.5),
                       SceneObject('bed_2', 'bed', 1.0, 1.0, 0.5, 5, 0.5), SceneObject('bed_5', 'bed', 2.0, 2.0, 0.5, 5, 0.5)])


@pytest.mark.parametrize('user,model,expected', [
    ('take me to the cart', 'GOAL:cart_1', 'cart'),                      # bare class: nearest, not the model's pick
    ('go to the beds', 'GOAL:bed_2', 'bed'),
    ('go to the vending machine', 'GOAL:vending_machine_2', 'vending machine'),
    ('go to cart_3', 'GOAL:cart_1', 'cart_3'),                           # named instance beats the model's mistake
    ('go to vending machine 2', 'GOAL:vending machine_1', 'vending machine_2'),
    ('go to the cart', 'GOAL:cart', 'cart'),                             # already a class
    ('go to bed 2', 'GOAL:bed_2', 'bed_2'),                              # the user gave a number: left alone
    ('go to the 2nd bed', 'GOAL:bed_2', 'bed_2'),
    ('go to the spaceship', 'GOAL:cart_1', 'cart_1'),                    # class not mentioned: the model's call stands
])
def test_goal_agrees_with_what_the_user_said(user, model, expected):
    g = _carts()
    assert reconcile_goal(parse_response(model, g.names()), user, g).target == expected


def test_only_goals_are_reconciled():
    g = _carts()
    for reply in ('MOVE:forward 2', 'STOP', 'I see a cart.'):
        cmd = parse_response(reply, g.names())
        assert reconcile_goal(cmd, 'take me to the cart', g) is cmd


def test_an_object_the_user_names_is_always_listed_even_beyond_the_cap():
    doors = [SceneObject(f'door_{i}', 'door', float(i), 0.0, 1.0, 5, 0.5) for i in range(1, 15)]
    g = SceneGraph(doors)
    plain = build_system_prompt(g, (0.0, 0.0, 0.0))
    assert 'door_13' not in plain and '+8 more' in plain
    named = build_system_prompt(g, (0.0, 0.0, 0.0), mentioned_objects('go to door_13', g))
    assert 'door_13 (13 m)' in named and 'door_14' not in named and '+7 more' in named
    assert 'door_13' in build_question_prompt(g, (0.0, 0.0, 0.0), mentioned_objects('where is door_13?', g))
    assert [o.id for o in mentioned_objects('take me to Door 3, not door_31', g)] == ['door_3']
    assert mentioned_objects('go to the door', g) == [] and mentioned_objects('hi', None) == []


def test_question_prompt_says_the_pose_is_the_robots_not_an_object_position():
    prompt = build_question_prompt(_carts(), (1.0, 2.0, 0.0))
    assert "ROBOT's own pose" in prompt and 'describe the picture' in prompt and 'Do not output commands' in prompt
