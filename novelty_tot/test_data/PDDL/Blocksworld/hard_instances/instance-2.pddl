

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b f)
(on c a)
(on d b)
(on e d)
(on f c)
(clear e)
)
(:goal
(and
(on b d)
(on d f)
(on e b))
)
)


