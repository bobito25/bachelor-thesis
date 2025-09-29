

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b c)
(on c a)
(on d b)
(ontable e)
(on f d)
(clear e)
(clear f)
)
(:goal
(and
(on a e)
(on b d)
(on c a)
(on e b))
)
)


