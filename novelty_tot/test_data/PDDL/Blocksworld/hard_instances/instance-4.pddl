

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b c)
(on c a)
(on d b)
(on e f)
(ontable f)
(clear d)
(clear e)
)
(:goal
(and
(on a c)
(on c e)
(on d a)
(on e b))
)
)


