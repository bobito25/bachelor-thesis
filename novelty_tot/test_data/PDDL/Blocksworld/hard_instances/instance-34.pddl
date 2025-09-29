

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a f)
(on b c)
(on c a)
(ontable d)
(on e b)
(ontable f)
(clear d)
(clear e)
)
(:goal
(and
(on a f)
(on c d)
(on d a)
(on e b))
)
)


